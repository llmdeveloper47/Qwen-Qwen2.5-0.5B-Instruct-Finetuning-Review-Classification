#!/usr/bin/env python3
"""
GPT Evaluation Script - Computes metrics from prediction results JSON

This script reads a prediction results JSON file and computes classification metrics
including accuracy, precision, recall, F1 scores, and generates detailed reports.

Usage:
    python evaluate_gpt.py
    python evaluate_gpt.py --input path/to/predictions.json
    python evaluate_gpt.py --output results/evaluation_report.json
"""

import os
import json
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

DEFAULT_INPUT = "dataset/final_sampled_data/prediction_result.json"
DEFAULT_OUTPUT = "results/gpt_evaluation/evaluation_report.json"


def load_predictions(input_path: str) -> dict:
    """Load predictions from JSON file"""
    print("="*70)
    print("Loading Prediction Results")
    print("="*70)
    print(f"Input file: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Model: {data.get('model', 'Unknown')}")
    print(f"Total samples: {data.get('num_samples', len(data['results']))}")
    print(f"Elapsed time: {data.get('elapsed_time', 'N/A')} seconds")
    print(f"Throughput: {data.get('throughput', 'N/A')} predictions/second")
    
    return data


def prepare_labels_and_predictions(results: list) -> tuple:
    """
    Extract true labels and predictions from results
    
    Returns:
        tuple: (true_labels, predictions, label2id, id2label)
    """
    print("\n" + "="*70)
    print("Preparing Labels and Predictions")
    print("="*70)
    
    true_labels_str = []
    predictions_str = []
    
    for result in results:
        true_labels_str.append(result['true_label'])
        predictions_str.append(result['gpt_raw_prediction'])
    
    # Get all unique labels from both true and predicted
    all_labels = sorted(set(true_labels_str + predictions_str))
    
    # Create label mappings
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for idx, label in enumerate(all_labels)}
    
    # Convert to numeric IDs
    true_labels = [label2id[label] for label in true_labels_str]
    predictions = []
    
    # Handle predictions that might not be in original label set (errors, etc.)
    for pred in predictions_str:
        if pred in label2id:
            predictions.append(label2id[pred])
        else:
            # If prediction is an error or unknown label, assign to special category
            # or default to first label
            if pred not in label2id:
                # Add to mapping if it's an error
                if 'ERROR' in pred and pred not in label2id:
                    label2id[pred] = len(label2id)
                    id2label[len(id2label)] = pred
                predictions.append(label2id[pred])
            else:
                predictions.append(label2id[pred])
    
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    print(f"Number of unique labels: {len(all_labels)}")
    print(f"Label distribution:")
    for label in sorted(set(true_labels_str)):
        count = true_labels_str.count(label)
        print(f"  {label}: {count} ({count/len(true_labels_str)*100:.2f}%)")
    
    return true_labels, predictions, label2id, id2label


def compute_metrics(true_labels: np.ndarray, predictions: np.ndarray, id2label: dict) -> dict:
    """Compute classification metrics"""
    print("\n" + "="*70)
    print("Computing Metrics")
    print("="*70)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    precision_macro = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, predictions, average='macro', zero_division=0)
    
    # Print results
    print(f"\nAccuracy:           {accuracy:.4f}")
    print(f"Balanced Accuracy:  {balanced_acc:.4f}")
    print(f"F1 Score (Macro):   {f1_macro:.4f}")
    print(f"F1 Score (Weighted):{f1_weighted:.4f}")
    print(f"Precision (Macro):  {precision_macro:.4f}")
    print(f"Recall (Macro):     {recall_macro:.4f}")
    
    # Classification report
    print("\n" + "="*70)
    print("Classification Report")
    print("="*70)
    
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    report = classification_report(
        true_labels,
        predictions,
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=False
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\n" + "="*70)
    print("Confusion Matrix")
    print("="*70)
    print("Rows: True labels | Columns: Predicted labels")
    print(f"\nLabels: {target_names}")
    print(cm)
    
    # Get detailed classification report as dict for saving
    report_dict = classification_report(
        true_labels,
        predictions,
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict
    }


def analyze_errors(results: list, label2id: dict) -> dict:
    """Analyze prediction errors"""
    print("\n" + "="*70)
    print("Error Analysis")
    print("="*70)
    
    errors = []
    error_types = {}
    
    for result in results:
        true_label = result['true_label']
        prediction = result['gpt_raw_prediction']
        
        if prediction != true_label:
            error_type = f"{true_label} → {prediction}"
            errors.append({
                'index': result['index'],
                'true_label': true_label,
                'prediction': prediction,
                'review_preview': result['review_text'][:100]
            })
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    # Sort errors by frequency
    sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTotal errors: {len(errors)}/{len(results)} ({len(errors)/len(results)*100:.2f}%)")
    print(f"\nTop 10 error patterns:")
    for error_pattern, count in sorted_errors[:10]:
        print(f"  {error_pattern}: {count} times")
    
    # Show a few example errors
    print(f"\nExample errors (first 5):")
    for error in errors[:5]:
        print(f"\n  Index {error['index']}:")
        print(f"    True: {error['true_label']}")
        print(f"    Pred: {error['prediction']}")
        print(f"    Text: {error['review_preview']}...")
    
    return {
        'total_errors': len(errors),
        'error_rate': len(errors)/len(results),
        'top_error_patterns': sorted_errors[:20],
        'sample_errors': errors[:10]
    }


def save_results(metrics: dict, error_analysis: dict, metadata: dict, output_path: str):
    """Save evaluation results to JSON"""
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)
    print(f"Output file: {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output_data = {
        'metadata': metadata,
        'metrics': metrics,
        'error_analysis': error_analysis
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("✓ Results saved successfully")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate GPT predictions from JSON results file"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=DEFAULT_INPUT,
        help=f'Path to prediction results JSON file (default: {DEFAULT_INPUT})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT,
        help=f'Path to output evaluation report (default: {DEFAULT_OUTPUT})'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    print("="*70)
    print("GPT Predictions Evaluation")
    print("="*70)
    
    # Load predictions
    data = load_predictions(args.input)
    results = data['results']
    
    # Prepare labels and predictions
    true_labels, predictions, label2id, id2label = prepare_labels_and_predictions(results)
    
    # Compute metrics
    metrics = compute_metrics(true_labels, predictions, id2label)
    
    # Analyze errors
    error_analysis = analyze_errors(results, label2id)
    
    # Prepare metadata
    metadata = {
        'input_file': args.input,
        'model': data.get('model', 'Unknown'),
        'num_samples': len(results),
        'elapsed_time': data.get('elapsed_time', 'N/A'),
        'throughput': data.get('throughput', 'N/A'),
        'num_labels': len(label2id)
    }
    
    # Save results
    save_results(metrics, error_analysis, metadata, args.output)
    
    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Error Rate: {error_analysis['error_rate']*100:.2f}%")
    print(f"\nResults saved to: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()
