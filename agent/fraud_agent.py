import pickle
import pandas as pd
from pathlib import Path
from preprocessing.feature_engineering import preprocess
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

BASE_DIR = Path(__file__).resolve().parent.parent  # parent of models/
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
RULES_PATH = BASE_DIR / "rag" / "docs" / "fraud_rules.txt"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(RULES_PATH) as f:
    rag_docs = f.read().splitlines()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedding_model.encode(rag_docs, convert_to_tensor=True)

# Load HF model for explanation
model_id = 'facebook/opt-125m'
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
llm = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

def compute_risk_score(user_df):
    user_df_preprocessed = preprocess(user_df)
    proba = model.predict_proba(user_df_preprocessed)[0][1]  # probability of fraud
    return proba

def make_decision(score):
    if score > 0.7:
        return "REJECT / FLAG"
    elif score > 0.4:
        return "MANUAL REVIEW"
    else:
        return "APPROVE"

def generate_explanation(user_df, score, decision):
    # Retrieve relevant RAG docs
    user_text = "Transaction features: " + str(user_df.to_dict(orient='records')[0])
    query_emb = embedding_model.encode(user_text, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, doc_embeddings, top_k=3)
    retrieved = [rag_docs[h['corpus_id']] for h in hits[0]]
    
    # Create LLM prompt
    prompt = f"""
    You are a fraud detection system.

    Transaction details:
    {user_text}

    Fraud probability: {score:.2f}
    
    Decision: {decision}

    Relevant rules:
    {retrieved}

    Explanation:
    This transaction is {decision} because
    """

    output = llm(
        prompt, 
        max_length=120, 
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False
        )
    return output[0]['generated_text']

def agent_decision(user_df):
    score = compute_risk_score(user_df)
    decision = make_decision(score)
    explanation = generate_explanation(user_df, score, decision)
    return score, decision, explanation