import json
import pandas as pd
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(pred, ref):
    embeddings = model.encode([pred, ref])
    return float(util.cos_sim(embeddings[0], embeddings[1])[0])

# Aggregates feedback, detects hallucinations (BERT F1 < 0.85), and exports a CSV summary
def analyze_logs(log_file="chatbot_logs.jsonl"):
    with open(log_file) as f:
        logs = [json.loads(line) for line in f]

    questions = [log["question"] for log in logs]
    responses = [log["response"] for log in logs]
    _, _, f1s = score(responses, questions, lang="en", rescale_with_baseline=True)

    data = []
    for i, log in enumerate(logs):
        f1 = f1s[i].item()
        sim = semantic_similarity(log["response"], log["question"])
        thumbs = log.get("feedback", {}).get("thumbs_up", False)
        data.append({
            "timestamp": log["timestamp"],
            "session_id": log.get("session_id", ""),
            "question": log["question"],
            "response": log["response"],
            "bert_f1": round(f1, 4),
            "cosine_sim": round(sim, 4),
            "latency": log.get("latency_sec", 0),
            "thumbs_up": thumbs,
            "comment": log.get("feedback", {}).get("comment", ""),
            "hallucination": f1 < 0.85
        })

    df = pd.DataFrame(data)
    df.to_csv("log_analysis_summary.csv", index=False)
    print(df.describe())

if __name__ == "__main__":
    analyze_logs()
