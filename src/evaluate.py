import json
from bert_score import score as bert_score
import time

def evaluate_model(chatbot_function, test_path):
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    results = []
    start_time = time.time()
    for entry in test_data:
        question = entry["question"]
        expected = entry["answer"]
        response = chatbot_function(question)

        # Compute BERTScore
        P, R, F1 = bert_score([response], [expected], lang="en", rescale_with_baseline=True)
        
        results.append({
            "question": question,
            "response": response,
            "expected": expected,
            "bert_f1": F1.item(),
            "match": "✅" if F1.item() > 0.85 else "❌"
        })

    avg_f1 = sum(r["bert_f1"] for r in results) / len(results)
    latency = (time.time() - start_time) / len(results)

    # Save results to file
    with open("evaluation_results.json", "w") as out:
        json.dump(results, out, indent=2)

    print("✅ Evaluation Summary")
    print(f"Average BERT F1: {avg_f1:.3f}")
    print(f"Average Latency: {latency:.2f} seconds per query")
