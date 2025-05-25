import json
import time
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

# Load necessary models
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Core Metrics
def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())

def bleu_score(pred, gold):
    return sentence_bleu([gold.split()], pred.split(), smoothing_function=SmoothingFunction().method1)

def cosine_similarity(pred, gold):
    embeddings = semantic_model.encode([pred, gold])
    return float(util.cos_sim(embeddings[0], embeddings[1])[0])

def compute_rouge(pred, gold):
    scores = rouge.score(gold, pred)
    return {
        "rouge1": round(scores['rouge1'].fmeasure, 4),
        "rouge2": round(scores['rouge2'].fmeasure, 4),
        "rougeL": round(scores['rougeL'].fmeasure, 4)
    }

# Quality/Heuristic Metrics
def compute_answer_length(text):
    return len(text.split())

# Evaluation driver
def evaluate_model(chatbot_function, test_path):
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    results = []
    total_latency = 0

    for entry in test_data:
        question = entry["question"]
        expected = entry["answer"]
        category = entry.get("category", "Uncategorized")

        start_time = time.time()
        response = chatbot_function(question)
        latency = time.time() - start_time
        total_latency += latency

        P, R, F1 = bert_score([response], [expected], lang="en", rescale_with_baseline=True)
        em = exact_match(response, expected)
        bleu = bleu_score(response, expected)
        cos = cosine_similarity(response, expected)
        rouge_scores = compute_rouge(response, expected)
        length = compute_answer_length(response)

        result = {
            "category": category,
            "question": question,
            "expected": expected,
            "response": response,
            "bert_f1": round(F1.item(), 4),
            "exact_match": em,
            "bleu_score": round(bleu, 4),
            "cosine_similarity": round(cos, 4),
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "answer_length": length,
            "latency_sec": round(latency, 2),
            
            # Placeholder advanced metrics (defined in commented statements below)
            "factual_consistency": "TODO",
            "citation_accuracy": "TODO",
            "medical_coverage": "TODO",
            "hallucination_flag": "TODO",
            "confidence_calibration": "TODO",
            "response_completeness": "TODO"
        }

        results.append(result)

    # Save to file
    with open("evaluation_results.json", "w") as out:
        json.dump(results, out, indent=2)

    print(f"\n📊 Evaluation complete on {len(results)} samples.")
    print(f"Average latency: {total_latency/len(results):.2f} seconds per query.")
    print("Results saved to evaluation_results.json")

# Demo run
if __name__ == "__main__":
    def dummy_chatbot(q): return "This is a placeholder answer."
    evaluate_model(dummy_chatbot, "data/sample_questions.json")

# Optional clinical specific metrics (if data available):

# def factual_consistency_score(predicted_answers, source_documents):
#     consistenry_scores = []
#         for pred, source in zip(predicted_answers, source_documents):
#             pred_doc = self.nlp(pred)
#             source_doc = self.nlp(source)
            
#             pred_entities = set([ent.text.lower() for ent in pred_doc.ents])
#             source_entities = set([ent.text.lower() for ent in source_doc.ents])
            
#             if len(pred_entities) == 0:
#                 consistency_scores.append(1.0)  # No entities claimed
#             else:
#                 overlap = len(pred_entities.intersection(source_entities))
#                 consistency = overlap / len(pred_entities)
#                 consistency_scores.append(consistency)
        
#         return np.mean(consistency_scores) 

# def citation_accuracy(predicted_answers, has_citations_ground_truth):
#     citation_pattern = r'(section|figure|table|appendix)\s+\d+|fig\.\s*\d+|ref\.\s*\d+'
#     predicted_citations = [1 if re.search(citation_pattern, answer.lower()) else 0 
#                               for answer in predicted_answers]
        
#     return {
#             'accuracy': accuracy_score(has_citations_ground_truth, predicted_citations),
#             'precision': precision_score(has_citations_ground_truth, predicted_citations),
#             'recall': recall_score(has_citations_ground_truth, predicted_citations),
#             'f1': f1_score(has_citations_ground_truth, predicted_citations)
#         }
       
# def medical_terminology_coverage(predicted_answers, medical_terms_list):
#         term_coverage_scores = []
        
#         for answer in predicted_answers:
#             answer_lower = answer.lower()
#             terms_found = sum(1 for term in medical_terms_list 
#                             if term.lower() in answer_lower)
#             coverage = terms_found / len(medical_terms_list) if medical_terms_list else 0
#             term_coverage_scores.append(coverage)
        
#         return {
#             'mean_coverage': np.mean(term_coverage_scores),
#             'median_coverage': np.median(term_coverage_scores)
#         }

 # def hallucination_rate(predicted_answers, source_documents, threshold=0.3):
        
 #        hallucination_scores = []
        
 #        for pred, source in zip(predicted_answers, source_documents):
 #            pred_embedding = self.model.encode([pred])
 #            source_embedding = self.model.encode([source])
            
 #            similarity = 1 - cosine(pred_embedding[0], source_embedding[0])
 #            is_hallucination = similarity < threshold
 #            hallucination_scores.append(is_hallucination)
        
 #        return {
 #            'hallucination_rate': np.mean(hallucination_scores),
 #            'similarity_threshold': threshold
 #        }

# def confidence_calibration(predicted_answers, confidence_scores, ground_truth_labels):

#         # Bin predictions by confidence and calculate accuracy in each bin
#         n_bins = 10
#         bin_boundaries = np.linspace(0, 1, n_bins + 1)
#         bin_lowers = bin_boundaries[:-1]
#         bin_uppers = bin_boundaries[1:]
        
#         bin_accuracies = []
#         bin_confidences = []
#         bin_counts = []
        
#         for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
#             in_bin = [(conf > bin_lower) and (conf <= bin_upper) 
#                      for conf in confidence_scores]
            
#             if any(in_bin):
#                 bin_accuracy = np.mean([gt for gt, in_b in zip(ground_truth_labels, in_bin) if in_b])
#                 bin_confidence = np.mean([conf for conf, in_b in zip(confidence_scores, in_bin) if in_b])
#                 bin_count = sum(in_bin)
                
#                 bin_accuracies.append(bin_accuracy)
#                 bin_confidences.append(bin_confidence)
#                 bin_counts.append(bin_count)
        
#         # Expected Calibration Error (ECE)
#         ece = sum([count * abs(acc - conf) for acc, conf, count 
#                   in zip(bin_accuracies, bin_confidences, bin_counts)])
#         ece /= sum(bin_counts) if bin_counts else 1
        
#         return {
#             'expected_calibration_error': ece,
#             'bin_accuracies': bin_accuracies,
#             'bin_confidences': bin_confidences,
#             'bin_counts': bin_counts
#         }

 # def response_completeness_score(predicted_answers, question_aspects, ground_truth_aspects=None):
  
 #        completeness_scores = []
        
 #        for i, (answer, aspects) in enumerate(zip(predicted_answers, question_aspects)):
 #            answer_lower = answer.lower()
            
 #            # Count how many aspects are addressed in the response
 #            aspects_covered = 0
 #            for aspect in aspects:
 #                # Simple keyword matching - could be enhanced with semantic matching
 #                if any(keyword.lower() in answer_lower for keyword in aspect.split()):
 #                    aspects_covered += 1
            
 #            completeness = aspects_covered / len(aspects) if aspects else 0
 #            completeness_scores.append(completeness)
        
 #        result = {
 #            'mean_completeness': np.mean(completeness_scores),
 #            'median_completeness': np.median(completeness_scores),
 #            'std_completeness': np.std(completeness_scores),
 #            'individual_scores': completeness_scores
 #        }
        
 #        # If ground truth aspects provided, compare coverage
 #        if ground_truth_aspects:
 #            gt_completeness_scores = []
 #            relative_completeness_scores = []
            
 #            for answer, gt_aspects in zip(predicted_answers, ground_truth_aspects):
 #                answer_lower = answer.lower()
 #                gt_aspects_covered = 0
                
 #                for aspect in gt_aspects:
 #                    if any(keyword.lower() in answer_lower for keyword in aspect.split()):
 #                        gt_aspects_covered += 1
                
 #                gt_completeness = gt_aspects_covered / len(gt_aspects) if gt_aspects else 0
 #                gt_completeness_scores.append(gt_completeness)
            
 #            result['ground_truth_completeness'] = {
 #                'mean': np.mean(gt_completeness_scores),
 #                'individual_scores': gt_completeness_scores
 #            }
        
 #        return result
