from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataset_utils import normalize_text

# Load the embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_embeddings(symptoms_list):
    """Convert all dataset symptoms into embeddings"""
    return model.encode(symptoms_list, convert_to_tensor=True)

def predict_condition(user_input, data, symptom_embeddings, top_k=3):
    """Return top_k matches with similarity scores"""
    user_input = normalize_text(user_input)
    if not user_input:
        return []

    user_emb = model.encode([user_input], convert_to_tensor=True)
    sims = cosine_similarity(user_emb.cpu().detach().numpy(),
                             symptom_embeddings.cpu().detach().numpy()).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        row = data.iloc[idx]
        results.append({
            "rank": rank,
            "name": row["Name"],
            "disease_code": row["Disease_Code"],
            "symptoms": row["Symptoms"],
            "treatment": row["Treatments"],
            "contagious": row["Contagious"],
            "chronic": row["Chronic"],
            "similarity": round(sims[idx]*100,1)
        })
    return results
