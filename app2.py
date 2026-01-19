import os
import re
import fitz
import nltk
import uuid
import torch
import spacy
import uvicorn
import logging
import warnings
import numpy as np

from typing import List, Dict
from typing import Dict, Optional
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import  HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore", message="`encoder_attention_mask` is deprecated")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup NLTK punkt tokenizer with local check
nltk_data_dir = os.getenv('NLTK_DATA_DIR', None)
if nltk_data_dir:
    nltk.data.path.append(nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("NLTK punkt not found, downloading...")
    nltk.download('punkt')


# Initialize CUDA if available
device = torch.device("cpu")
logger.info(f"Using device: {device}")



# Load DistilBERT model and tokenizer for semantic similarity
pipeline_device = -1

model_name = "llmware/industry-bert-insurance-v0.1"

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load embedding model and move to correct device
    embedding_model = AutoModel.from_pretrained(model_name).to(device)

    # Load classification model and move to correct device
    classification_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    # Setup a feature-extraction pipeline on the correct device
    embedding_pipeline = pipeline(
        "feature-extraction",
        model=embedding_model,
        tokenizer=tokenizer,
        framework="pt",
        device=pipeline_device
    )

   
except Exception as e:
    logger.error(f"Failed to load model or tokenizer: {str(e)}")
    raise Exception("Model loading failed. Ensure transformers and PyTorch are correctly installed.")



# Pre-compiled regex patterns for fields
AGE_PATTERN = re.compile(r'\b(\d{1,3})\s*(?:years? old|yr old|y/o|yrs old)?\b', re.I)
DURATION_PATTERN = re.compile(r'(?:policy for|coverage for|duration of|valid for)\s*([\w\s]+)', re.I)
PROCEDURE_PATTERN = re.compile(r'(?:undergo|scheduled for|require|need(?:s)?|plan(?:ned)? for)\s*([\w\s\-]+?)(?:[.,]|$)', re.I)
CASE_TYPE_PATTERN = re.compile(r'\b(emergency|planned)\b', re.I)
INJURY_PATTERN = re.compile(r'(?:injury(?: of| to)?|accident|fracture|wound)\s*([\w\s\-]+?)(?:[.,]|$)', re.I)
ILLNESS_PATTERN = re.compile(r'(?:diagnosed|illness(?: of| is)?|disease|infection)\s*([\w\s\-]+?)(?:[.,]|$)', re.I)


# FastAPI app instance
app = FastAPI(title="Insurance Query Processing API")

# Embedded HTML content
with open("interface.html","r",encoding="utf-8") as inf:
    HTML_CONTENT = inf.read()

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Serve the main HTML interface"""
    return HTMLResponse(content=HTML_CONTENT)


SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+')
MIN_CLAUSE_LEN = 15
MAX_CLAUSE_LEN = 150

async def parse_document_from_pdf(file: UploadFile) -> List[Dict]:
    
    try:
        content = await file.read()  # read PDF bytes asynchronously
        clauses = []

        with fitz.open(stream=content, filetype='pdf') as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text:
                    sentences = SENTENCE_SPLIT_REGEX.split(text.strip())
                    for sentence in sentences:
                        sentence_clean = sentence.strip()
                        length = len(sentence_clean)
                        # Filter out too short or too long sentences
                        if sentence_clean and MIN_CLAUSE_LEN < length <= MAX_CLAUSE_LEN:
                            clauses.append({
                                "id": str(uuid.uuid4()),
                                "text": sentence_clean,
                                "page": str(page_num + 1)
                            })

        if not clauses:
            raise HTTPException(status_code=400, detail="No text extracted from PDF or no valid clauses found")

        logger.info(f"Extracted {len(clauses)} clauses from PDF.")
        # print(str(clauses))
        return clauses

    except Exception as e:
        logger.error(f"Error parsing PDF: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to parse uploaded PDF document")

    finally:
        await file.close()



# Load transformer spaCy pipeline with better NER (load once globally)
try:
    nlp_spacy_trf = spacy.load("en_core_web_trf")
except Exception:
    # fallback to small model if transformer is not available
    import warnings
    warnings.warn("Transformer model 'en_core_web_trf' not found, falling back to 'en_core_web_sm'.")
    nlp_spacy_trf = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)

# Assume global compiled regex patterns: AGE_PATTERN, DURATION_PATTERN, etc. exist

def extract_first_entity(doc, labels):
    """Utility: returns first entity text matching any label in labels, or None"""
    for ent in doc.ents:
        if ent.label_ in labels and ent.text.strip():
            return ent.text.strip()
    return None

def safe_int_cast(val, default=None):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def parse_query(query: str) -> Dict:
    parsed = {
        "name": None,
        "age": None,
        "procedure": None,
        "location": None,
        "policy_duration": None,
        "description of injury": None,
        "description of illness": None,
        "description of hospital": None,
        "type of treatment": None,
        "planned or emergency case": None,
        "injury": None,
    }

    if not query or not isinstance(query, str):
        logger.warning("Empty or invalid query input")
        return parsed

    query_lower = query.lower()

    # Use transformer model or fallback
    doc = nlp_spacy_trf(query_lower)

    # 1. Name: Prioritize PERSON, fallback to pattern-based extraction (if needed)
    name = extract_first_entity(doc, {"PERSON"})
    parsed["name"] = name

    # 2. Age: regex match first, else NER numeric detection (CARDINAL or QUANTITY)
    age_match = AGE_PATTERN.search(query_lower)
    if age_match:
        parsed["age"] = safe_int_cast(age_match.group(1))
    else:
        age_entity = extract_first_entity(doc, {"CARDINAL", "QUANTITY"})
        parsed["age"] = safe_int_cast(age_entity)

    # 3. Location: GPE or LOC
    parsed["location"] = extract_first_entity(doc, {"GPE", "LOC"})

    # 4. Policy duration: regex or time/date entities
    duration_match = DURATION_PATTERN.search(query_lower)
    if duration_match and duration_match.group(1).strip():
        parsed["policy_duration"] = duration_match.group(1).strip()
    else:
        parsed["policy_duration"] = extract_first_entity(doc, {"DATE", "TIME"})

    # 5. Procedure: regex first, else spaCy EVENT/PROCEDURE (use Event if Procedure not in model)
    procedure_match = PROCEDURE_PATTERN.search(query_lower)
    if procedure_match and procedure_match.group(1).strip():
        parsed["procedure"] = procedure_match.group(1).strip()
    else:
        parsed["procedure"] = extract_first_entity(doc, {"PROCEDURE", "EVENT", "WORK_OF_ART"})  # WORK_OF_ART often captures procedures

    # 6. Planned or emergency
    case_type_match = CASE_TYPE_PATTERN.search(query_lower)
    if case_type_match and case_type_match.group(1).strip():
        parsed["planned or emergency case"] = case_type_match.group(1).capitalize()

    # 7. Injury description and injury term
    injury_match = INJURY_PATTERN.search(query_lower)
    if injury_match:
        parsed["description of injury"] = injury_match.group(0).strip()
        parsed["injury"] = injury_match.group(1).strip()

    # 8. Illness description
    illness_match = ILLNESS_PATTERN.search(query_lower)
    if illness_match:
        parsed["description of illness"] = illness_match.group(0).strip()

    # 9. Hospital: ORG containing hospital-related keywords, fallback to manual keyword search in query
    hospitals = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG" and any(k in ent.text.lower() for k in ("hospital", "clinic", "nursing"))]
    if hospitals:
        parsed["description of hospital"] = hospitals[0]
    else:
        # fallback to keyword search in raw query
        for keyword in ("hospital", "clinic", "nursing"):
            if keyword in query_lower:
                parsed["description of hospital"] = keyword
                break

    # 10. Type of treatment: explicit keywords or phrase detection
    treatment_keywords = ["surgery", "medical management", "therapy", "medication", "consultation", "treatment", "chemotherapy", "radiation"]
    for keyword in treatment_keywords:
        if keyword in query_lower:
            parsed["type of treatment"] = keyword
            break

    return parsed





def precompute_clause_embeddings(clauses: List[Dict]) -> np.ndarray:

    try:
        texts = [clause["text"] for clause in clauses]
        embeddings = get_embedding(texts)
        return embeddings
    except Exception as e:
        logger.error(f"Failed to precompute clause embeddings: {e}")
        raise





def get_embedding(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Compute embeddings for a single text string or a list of texts using the loaded transformer model.
    
    Args:
        texts (str or List[str]): Input text or list of input texts to embed.
    
    Returns:
        np.ndarray: 2D array of embeddings with shape (num_texts, embedding_dim) when input is list.
                    1D array with shape (embedding_dim,) if input is a single string.
    """
    
    if isinstance(texts, str):
        texts = [texts]  # convert single string to list for batch processing

    # Tokenize batch of texts with padding and truncation
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,  # ensure max length to handle long inputs safely
    ).to(device)

    with torch.no_grad():
        outputs = embedding_model(**inputs)
        # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_dim)

        # Mean pooling on token embeddings while excluding padding tokens
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch_size, seq_len, 1)
        masked_embeddings = outputs.last_hidden_state * attention_mask  # zero out pad tokens
        
        sum_embeddings = masked_embeddings.sum(dim=1)  # sum over seq_len
        lengths = attention_mask.sum(dim=1)  # number of valid tokens per sample

        # Avoid division by zero
        lengths = lengths.clamp(min=1e-9)

        embeddings = sum_embeddings / lengths  # mean pooling normalized

    embeddings = embeddings.cpu().numpy()

    # If original input was single string, return 1D array for convenience
    if len(texts) == 1:
        return embeddings[0]
    return embeddings



def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps=1e-10) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < eps or norm_b < eps:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def search_clauses(
    query: str,
    clauses: List[Dict],
    top_k: int = 3,
    clause_embeddings: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Return top_k most relevant clauses by semantic similarity to the query.

    Args:
        query: User query string
        clauses: List of clause dicts (each has at least a "text" key)
        top_k: Number of top results to return
        clause_embeddings: Optional precomputed embeddings (shape: num_clauses x emb_dim)

    Returns:
        List of clause dicts augmented with 'similarity' float score key.
    """
    try:
        # Embed the query
        query_embedding = get_embedding(query).flatten()

        # Precompute clause embeddings if not provided
        if clause_embeddings is None:
            texts = [clause["text"] for clause in clauses]
            clause_embeddings = get_embedding(texts)

        # Ensure dimensions
        assert clause_embeddings.ndim == 2, "Clause embeddings must be 2D array"

        # Compute cosine similarities in batch
        query_norm = np.linalg.norm(query_embedding)
        if query_norm < 1e-10:
            logger.warning("Query embedding norm is near zero; no results returned.")
            return []

        clause_norms = np.linalg.norm(clause_embeddings, axis=1)
        clause_norms = np.where(clause_norms < 1e-10, 1e-10, clause_norms)  # prevent div by zero

        dot_products = clause_embeddings @ query_embedding
        similarities = dot_products / (clause_norms * query_norm)

        # Get top-k indices sorted descending
        top_indices = np.argsort(similarities)[::-1][:top_k]

        top_clauses = []
        for idx in top_indices:
            clause = clauses[idx].copy()
            clause["similarity"] = float(similarities[idx])
            top_clauses.append(clause)

        return top_clauses

    except Exception as e:
        logger.error(f"Error during search_clauses: {str(e)}")
        raise RuntimeError("Semantic search failure") from e



def classify_clause_coverage(query: str, clause_text: str) -> Dict:
    """
    Classify whether the given policy clause covers the user query scenario.

    Returns:
        Dict with keys:
            - 'covered' (bool): True if model predicts coverage, else False.
            - 'score' (float): Confidence probability of coverage (class 1).
    """
    inputs = tokenizer(
        query,
        clause_text,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = classification_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        covered_prob = float(probabilities[1])  # probability for 'covered' class (index 1)
        covered_label = bool(np.argmax(probabilities) == 1)

    return {"covered": covered_label, "score": covered_prob}

def evaluate_clauses(
    query: str,
    parsed_query: Dict,
    clauses: List[Dict],
    coverage_threshold: float = 0.5,
) -> Dict:
    """
    Evaluate user query coverage against relevant policy clauses using ML classifier.

    Args:
        query (str): Raw user query text.
        parsed_query (Dict): Parsed structured info from query (e.g., age, procedure).
        clauses (List[Dict]): List of relevant clauses dicts, each containing 'text' and optionally 'page'.
        coverage_threshold (float): Minimum confidence to accept 'covered' classification.

    Returns:
        Dict: {
            'decision': 'approved' or 'rejected',
            'amount': int coverage amount,
            'justification': List of dicts with clause text, page, reason, and impact.
        }
    """
    decision = {
        "decision": "rejected",
        "amount": 0,
        "justification": []
    }

    coverage_results = []

    for clause in clauses:
        result = classify_clause_coverage(query, clause["text"])
        coverage_results.append({
            "clause_text": clause["text"],
            "page": clause.get("page", "N/A"),
            "covered": result["covered"],
            "score": result["score"]
        })

    # Filter only clauses confidently predicting coverage
    positive_matches = [r for r in coverage_results if r["covered"] and r["score"] >= coverage_threshold]

    if positive_matches:
        # Best match clause in terms of confidence
        best_match = max(positive_matches, key=lambda x: x["score"])

        clause_text_lower = best_match["clause_text"].lower()
        procedure = parsed_query.get("procedure", "")
        procedure_lower = procedure.lower() if procedure else ""
        age = parsed_query.get("age")

        # Convert age safely to integer if possible
        try:
            age_int = int(age)
        except (TypeError, ValueError):
            age_int = None

        # Determine base coverage amount heuristically
        if "hospital" in clause_text_lower or "admission" in clause_text_lower or procedure_lower == "hospitalization":
            base_amount = 750000
        elif "surgery" in clause_text_lower or procedure_lower == "surgery":
            base_amount = 500000
        elif "trip cancellation" in query.lower():
            base_amount = 250000
        else:
            base_amount = 300000  # fallback coverage amount

        # Apply age-based adjustment (senior citizens)
        if age_int and age_int > 60:
            final_amount = int(base_amount * 0.8)  # 20% discount
        else:
            final_amount = base_amount

        decision.update({
            "decision": "approved",
            "amount": final_amount,
            "justification": [{
                "clause": best_match["clause_text"],
                "page": best_match["page"],
                "reason": f"Clause matched with high confidence ({best_match['score']:.3f}) confirms coverage for the query.",
                "impact": f"Coverage amount set to ₹{final_amount:,} in accordance with policy terms and patient age."
            }]
        })

    else:
        # If no confident coverage match found
        decision.update({
            "decision": "rejected",
            "amount": 0,
            "justification": [{
                "clause": "Policy General Review",
                "page": "N/A",
                "reason": "No policy clauses sufficiently match the query scenario with high confidence.",
                "impact": "Claim rejected or requires manual review with additional documentation."
            }]
        })
    return decision



@app.post("/process_query")
async def process_query(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    # Step 0: Validate input file
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        logger.info(f"Received query: '{query}' with file: {file.filename}")

        # Step 1: Parse PDF and extract clauses
        clauses = await parse_document_from_pdf(file)
        if not clauses:
            raise HTTPException(status_code=400, detail="No extractable text found in PDF")

        # Step 2: Precompute or retrieve cached clause embeddings (optimize repeated use)
        clause_embeddings = precompute_clause_embeddings(clauses)

        # Step 3: Parse structured information from user query
        parsed_query = parse_query(query)

        # Step 4: Retrieve relevant clauses semantically
        relevant_clauses = search_clauses(
            query,
            clauses,
            top_k=5,  # Can tune this number as needed
            clause_embeddings=clause_embeddings
        )
        if not relevant_clauses:
            logger.warning("No relevant clauses found for the query")

        # Step 5: Evaluate coverage decision via ML classifier
        decision = evaluate_clauses(query, parsed_query, relevant_clauses)

        # Step 6: Construct and return response JSON
        response_payload = {
            "query": query,
            "parsed_query": parsed_query,
            "decision": decision.get("decision", "rejected"),
            "coverage_amount": decision.get("amount", 0),
            "justification": decision.get("justification", [])
        }

        logger.info(f"Query processed successfully. Decision: {response_payload['decision']}")

        return JSONResponse(content=response_payload)

    except HTTPException:
        # Let FastAPI handle HTTPExceptions as is
        raise

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error: Failed to process query")




# New POST handler to process form submissions asynchronously and return JSON
@app.post("/submit_query")
async def submit_query(query: str = Form(...), file: UploadFile = File(...)):
    """
    Endpoint to process user query with uploaded PDF and return JSON result.
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            return JSONResponse(status_code=400, content={"error": "Only PDF files are supported"})

        # Extract clauses from PDF
        clauses = await parse_document_from_pdf(file)

        # Precompute clause embeddings once per uploaded document for faster search
        clause_embeddings = precompute_clause_embeddings(clauses)

        # Parse the user query for structured info
        parsed_query = parse_query(query)

        # Search for relevant clauses using semantic similarity
        relevant_clauses = search_clauses(query, clauses, top_k=5, clause_embeddings=clause_embeddings)

        # Evaluate coverage decision using ML model over relevant clauses
        decision = evaluate_clauses(query, parsed_query, relevant_clauses)

        response = {
            "query": query,
            "parsed_query": parsed_query,
            "decision": decision.get("decision", "rejected"),
            "coverage_amount": decision.get("amount", 0),
            "amount": decision.get("amount", 0),
            "justification": decision.get("justification", [])
        }
        # print(str(response))
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error processing submitted query: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error. Please try again later."})













# # Uncomment to run locally:
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000 )
    
# http://localhost:8000/