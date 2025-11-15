#!/usr/bin/env python3
"""
GPT Predictions Script with Concurrent Processing
Fast prediction testing on test dataset with concurrent API calls
"""

import os
import json
import asyncio
import time
import argparse
from typing import Dict, List, Tuple, Optional
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

DATASET_PATH = "dataset/final_sampled_data/test_df.csv"
SYSTEM_PROMPT_PATH = "src/system_prompts/system_prompt_gpt.txt"
OUTPUT_PATH = "dataset/final_sampled_data/prediction_result_gpt5.json"


def load_system_prompt(prompt_path: str) -> str:
    """Load the system prompt from file"""
    abs_path = os.path.abspath(prompt_path)
    print(f"  Absolute path: {abs_path}")
    print(f"  File exists: {os.path.exists(abs_path)}")
    print(f"  File size: {os.path.getsize(abs_path) if os.path.exists(abs_path) else 'N/A'} bytes")
    
    with open(abs_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"  Content length: {len(content)} characters")
        print(f"  First 100 chars: {content[:100] if content else '(empty)'}")
        return content


def load_test_data(dataset_path: str, n_samples: Optional[int] = None) -> pd.DataFrame:
    """Load test dataset with optional sample limit"""
    df = pd.read_csv(dataset_path)
    
    if n_samples:
        print(f"Loading first {n_samples} samples from: {dataset_path}")
        df_sample = df.head(n_samples)
        print(f"Loaded {len(df_sample)} samples")
    else:
        print(f"Loading entire dataset from: {dataset_path}")
        df_sample = df
        print(f"Loaded {len(df_sample):,} samples")
    
    return df_sample


async def get_prediction_async(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    review_text: str,
    true_label: str,
    index: int,
    semaphore: asyncio.Semaphore,
    retry_count: int = 3,
    timeout: float = 60.0
) -> Dict:
    """
    Get prediction from GPT model with concurrent processing and return detailed results
    """
    async with semaphore:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Review: {review_text}"}
        ]
        
        # Determine which parameters to use based on model
        # GPT-5 and newer models use max_completion_tokens and don't support temperature=0.0
        # GPT-4 and earlier use max_tokens and support temperature=0.0
        use_new_api = 'gpt-5' in model_name.lower() or 'o1' in model_name.lower()
        
        for attempt in range(retry_count):
            try:
                # Prepare API parameters
                api_params = {
                    "model": model_name,
                    "messages": messages,
                }
                
                # GPT-5 doesn't support temperature=0.0, only default (1)
                # GPT-4 and earlier support temperature=0.0
                if not use_new_api:
                    api_params["temperature"] = 0.0
                
                # Use appropriate token parameter
                # GPT-5 may need more tokens, so use a higher limit
                if use_new_api:
                    api_params["max_completion_tokens"] = 200
                else:
                    api_params["max_tokens"] = 50
                
                response = await asyncio.wait_for(
                    client.chat.completions.create(**api_params),
                    timeout=timeout
                )
                
                raw_prediction = response.choices[0].message.content.strip()
                
                return {
                    "index": index,
                    "review_text": review_text,
                    "true_label": true_label,
                    "gpt_raw_prediction": raw_prediction
                }
                
            except asyncio.TimeoutError:
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {
                        "index": index,
                        "review_text": review_text,
                        "true_label": true_label,
                        "gpt_raw_prediction": "ERROR_TIMEOUT"
                    }
                    
            except Exception as e:
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {
                        "index": index,
                        "review_text": review_text,
                        "true_label": true_label,
                        "gpt_raw_prediction": f"ERROR: {str(e)}"
                    }


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="GPT Predictions with concurrent processing"
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all samples)'
    )
    
    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=5,
        help='Maximum number of concurrent API requests (default: 5)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5',
        help='OpenAI model to use (default: gpt-4o-mini)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=DATASET_PATH,
        help=f'Path to dataset CSV file (default: {DATASET_PATH})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_PATH,
        help=f'Path to output JSON file (default: {OUTPUT_PATH})'
    )
    
    return parser.parse_args()


async def main():
    """Main function to test predictions with concurrent processing"""
    # Parse arguments
    args = parse_arguments()
    
    print("="*70)
    print("GPT Prediction Script (Concurrent Processing)")
    print("="*70)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\nError: OPENAI_API_KEY not found in environment variables.")
        return
    
    # Load system prompt
    print(f"\nLoading system prompt from: {SYSTEM_PROMPT_PATH}")
    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)
    print(f"System prompt loaded ({len(system_prompt)} characters)")
    
    # Load test data
    df = load_test_data(args.dataset, n_samples=args.num_samples)
    
    # Get all unique labels for reference
    all_labels = sorted(df['labels'].unique())
    print(f"\nUnique labels in sample: {len(all_labels)}")
    for label in all_labels:
        print(f"  - {label}")
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=api_key)
    model_name = args.model
    print(f"\nUsing model: {model_name}")
    
    # Set concurrency
    max_concurrent = args.max_concurrent
    print(f"Max concurrent requests: {max_concurrent}")
    semaphore = asyncio.Semaphore(max_concurrent)
    
    print(f"\n{'='*70}")
    print("Getting predictions with concurrent processing...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Create tasks for all predictions
    tasks = []
    for idx, row in df.iterrows():
        task = get_prediction_async(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            review_text=row['text'],
            true_label=row['labels'],
            index=int(idx),
            semaphore=semaphore
        )
        tasks.append(task)
    
    # Run all tasks concurrently with progress bar
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        result = await coro
        results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # Sort results by index to maintain order
    results.sort(key=lambda x: x['index'])
    
    print(f"\nCompleted {len(results)} predictions in {elapsed_time:.2f} seconds")
    print(f"Average time per prediction: {elapsed_time/len(results):.3f} seconds")
    print(f"Throughput: {len(results)/elapsed_time:.2f} predictions/second")
    
    # Print individual results (only first 10 for large datasets)
    show_limit = min(10, len(results))
    print(f"\n{'='*70}")
    print(f"Individual Results (showing first {show_limit}):")
    print(f"{'='*70}\n")
    
    for r in results[:show_limit]:
        print(f"Review {r['index'] + 1}/{len(results)}:")
        print(f"  True label: {r['true_label']}")
        print(f"  Review preview: {r['review_text'][:100]}...")
        print(f"  GPT prediction: '{r['gpt_raw_prediction']}'")
        
        # Check if it's a match
        if r['gpt_raw_prediction'] == r['true_label']:
            print(f"  ✓ Exact match!")
        elif 'ERROR' in r['gpt_raw_prediction']:
            print(f"  ✗ Error occurred")
        else:
            print(f"  ✗ Mismatch")
        print()
    
    if len(results) > show_limit:
        print(f"... and {len(results) - show_limit} more results")
    
    # Save results to JSON
    print(f"\n{'='*70}")
    print(f"Saving results to: {args.output}")
    print(f"{'='*70}")
    
    output_data = {
        "model": model_name,
        "num_samples": len(results),
        "max_concurrent": max_concurrent,
        "elapsed_time": elapsed_time,
        "throughput": len(results)/elapsed_time,
        "system_prompt_path": SYSTEM_PROMPT_PATH,
        "dataset_path": args.dataset,
        "results": results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Results saved successfully")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    # Calculate matches and errors
    exact_matches = sum(1 for r in results if r['gpt_raw_prediction'] == r['true_label'])
    errors = sum(1 for r in results if 'ERROR' in r['gpt_raw_prediction'])
    successful = len(results) - errors
    
    print(f"Total samples: {len(results)}")
    print(f"Successful predictions: {successful}/{len(results)}")
    print(f"Exact matches: {exact_matches}/{len(results)} ({exact_matches/len(results)*100:.1f}%)")
    print(f"Errors: {errors}/{len(results)}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {len(results)/elapsed_time:.2f} predictions/second")
    
    # Show examples of mismatches
    print(f"\n{'='*70}")
    print("MISMATCH EXAMPLES (if any)")
    print(f"{'='*70}")
    
    mismatch_count = 0
    for r in results:
        if 'ERROR' not in r['gpt_raw_prediction'] and r['gpt_raw_prediction'] != r['true_label']:
            if mismatch_count < 5:  # Show only first 5 mismatches
                print(f"\nReview {r['index'] + 1}:")
                print(f"  True label: '{r['true_label']}'")
                print(f"  GPT returned: '{r['gpt_raw_prediction']}'")
            mismatch_count += 1
    
    if mismatch_count > 5:
        print(f"\n... and {mismatch_count - 5} more mismatches")
    
    print(f"\n{'='*70}")
    print(f"Complete! Check {args.output} for full results")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())

