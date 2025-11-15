#!/usr/bin/env python3
"""
Convert CSV Files to HuggingFace DatasetDict Format and Upload to Hub

This script converts training, validation, and test CSV files to HuggingFace
DatasetDict format and optionally uploads to HuggingFace Hub.

Usage:
    python convert_to_huggingface_dataset.py --upload --repo_name your-username/dataset-name

Requirements:
    pip install datasets pandas huggingface_hub
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi


def load_and_validate_csv(csv_file: str, split_name: str) -> pd.DataFrame:
    """
    Load and validate CSV file
    
    Args:
        csv_file: Path to CSV file
        split_name: Name of split (train/test/val) for logging
        
    Returns:
        pd.DataFrame: Loaded and validated dataframe
    """
    print(f"\n{'='*70}")
    print(f"Loading {split_name} set: {csv_file}")
    print(f"{'='*70}")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{split_name} file not found: {csv_file}")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Remove index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Check for text/labels columns and rename to query/label
    if 'text' in df.columns and 'labels' in df.columns:
        df = df.rename(columns={'text': 'query', 'labels': 'label'})
    elif 'query' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"{split_name} CSV must have 'text'/'labels' or 'query'/'label' columns. Found: {df.columns.tolist()}")
    
    # Keep only query and label columns
    df = df[['query', 'label']].copy()
    
    # Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates(subset=['query', 'label'])
    duplicates_removed = original_len - len(df)
    
    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate entries")
    
    # Remove NaN values
    df = df.dropna()
    
    print(f"  Total samples: {len(df):,}")
    print(f"  Unique labels: {df['label'].nunique()}")
    
    # Show label distribution
    label_counts = df['label'].value_counts()
    print(f"\n  Label distribution (top 10):")
    for label, count in label_counts.head(10).items():
        print(f"    {label}: {count}")
    
    return df


def convert_to_huggingface_dataset(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    output_dir: str,
    validate_splits: bool = True
) -> DatasetDict:
    """
    Convert CSV files to HuggingFace DatasetDict
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        output_dir: Directory to save the DatasetDict
        validate_splits: Whether to validate label distribution
        
    Returns:
        DatasetDict: HuggingFace dataset with train/validation/test splits
    """
    print("="*70)
    print("Converting CSV Files to HuggingFace DatasetDict")
    print("="*70)
    
    # Load and validate datasets
    train_df = load_and_validate_csv(train_csv, "Train")
    val_df = load_and_validate_csv(val_csv, "Validation")
    test_df = load_and_validate_csv(test_csv, "Test")
    
    # Convert pandas DataFrames to HuggingFace Datasets
    print(f"\n{'='*70}")
    print("Converting to HuggingFace Dataset format...")
    print(f"{'='*70}")
    
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    
    print(f"  Train dataset created: {len(train_dataset)} samples")
    print(f"  Validation dataset created: {len(val_dataset)} samples")
    print(f"  Test dataset created: {len(test_dataset)} samples")
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    print(f"\n{'='*70}")
    print("DatasetDict Summary")
    print(f"{'='*70}")
    print(f"Splits: {list(dataset_dict.keys())}")
    print(f"Train samples: {len(dataset_dict['train']):,}")
    print(f"Validation samples: {len(dataset_dict['validation']):,}")
    print(f"Test samples: {len(dataset_dict['test']):,}")
    print(f"Total samples: {len(dataset_dict['train']) + len(dataset_dict['validation']) + len(dataset_dict['test']):,}")
    print(f"\nFeatures: {dataset_dict['train'].features}")
    
    # Validate label consistency
    if validate_splits:
        print(f"\n{'='*70}")
        print("Label Distribution Validation")
        print(f"{'='*70}")
        
        train_labels = set(dataset_dict['train']['label'])
        val_labels = set(dataset_dict['validation']['label'])
        test_labels = set(dataset_dict['test']['label'])
        
        print(f"Unique labels in train: {len(train_labels)}")
        print(f"Unique labels in validation: {len(val_labels)}")
        print(f"Unique labels in test: {len(test_labels)}")
        
        # Check for labels in validation but not in train
        val_only_labels = val_labels - train_labels
        if val_only_labels:
            print(f"\n  Warning: Labels in validation but not in train: {len(val_only_labels)}")
        else:
            print(f"  All validation labels are present in train set")
        
        # Check for labels in test but not in train
        test_only_labels = test_labels - train_labels
        if test_only_labels:
            print(f"  Warning: Labels in test but not in train: {len(test_only_labels)}")
        else:
            print(f"  All test labels are present in train set")
    
    # Save to disk
    if output_dir:
        print(f"\n{'='*70}")
        print(f"Saving DatasetDict to: {output_dir}")
        print(f"{'='*70}")
        
        os.makedirs(output_dir, exist_ok=True)
        dataset_dict.save_to_disk(output_dir)
        print(f"  DatasetDict saved successfully!")
    
    return dataset_dict


def upload_to_hub(dataset_dict: DatasetDict, repo_name: str, token: str = None):
    """
    Upload dataset to HuggingFace Hub
    
    Args:
        dataset_dict: DatasetDict to upload
        repo_name: Repository name (username/dataset-name)
        token: HuggingFace token (if None, will prompt for login)
    """
    print(f"\n{'='*70}")
    print("Uploading to HuggingFace Hub")
    print(f"{'='*70}")
    
    # Login to HuggingFace
    if token:
        login(token=token)
        print("  Logged in with provided token")
    else:
        print("  Please provide your HuggingFace token:")
        login()
    
    # Upload dataset
    print(f"\n  Uploading to: {repo_name}")
    print("  This may take a few minutes...")
    
    dataset_dict.push_to_hub(repo_name, private=False)
    
    print(f"\n  Successfully uploaded to HuggingFace Hub!")
    print(f"  View your dataset at: https://huggingface.co/datasets/{repo_name}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert CSV files to HuggingFace DatasetDict format and upload to Hub"
    )
    
    parser.add_argument(
        '--train_csv',
        type=str,
        default='dataset/final_sampled_data/train_df.csv',
        help='Path to training CSV'
    )
    
    parser.add_argument(
        '--val_csv',
        type=str,
        default='dataset/final_sampled_data/val_df.csv',
        help='Path to validation CSV'
    )
    
    parser.add_argument(
        '--test_csv',
        type=str,
        default='dataset/final_sampled_data/test_df.csv',
        help='Path to test CSV'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset/final_sampled_data/huggingface',
        help='Output directory for HuggingFace dataset'
    )
    
    parser.add_argument(
        '--no_validation',
        action='store_true',
        help='Skip label distribution validation'
    )
    
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload dataset to HuggingFace Hub'
    )
    
    parser.add_argument(
        '--repo_name',
        type=str,
        help='HuggingFace repository name (username/dataset-name)'
    )
    
    parser.add_argument(
        '--token',
        type=str,
        help='HuggingFace token (optional, will prompt if not provided)'
    )
    
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Skip saving to disk (only upload)'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    print("="*70)
    print("CSV to HuggingFace DatasetDict Converter")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Train CSV: {args.train_csv}")
    print(f"  Validation CSV: {args.val_csv}")
    print(f"  Test CSV: {args.test_csv}")
    print(f"  Output directory: {args.output_dir}")
    if args.upload:
        print(f"  Upload to Hub: Yes")
        print(f"  Repository: {args.repo_name}")
    
    # Check if files exist
    if not os.path.exists(args.train_csv):
        print(f"\nError: Train CSV not found: {args.train_csv}")
        return
    
    if not os.path.exists(args.val_csv):
        print(f"\nError: Validation CSV not found: {args.val_csv}")
        return
    
    if not os.path.exists(args.test_csv):
        print(f"\nError: Test CSV not found: {args.test_csv}")
        return
    
    # Convert to HuggingFace format
    try:
        output_dir = None if args.no_save else args.output_dir
        dataset_dict = convert_to_huggingface_dataset(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            output_dir=output_dir,
            validate_splits=not args.no_validation
        )
        
        # Upload to HuggingFace Hub if requested
        if args.upload:
            if not args.repo_name:
                print(f"\nError: --repo_name is required when using --upload")
                return
            
            upload_to_hub(dataset_dict, args.repo_name, args.token)
        
        print(f"\n{'='*70}")
        print("CONVERSION COMPLETE")
        print(f"{'='*70}")
        
        if not args.no_save:
            print(f"\nDataset saved at: {os.path.abspath(args.output_dir)}")
            print(f"\nTo load:")
            print(f"from datasets import load_from_disk")
            print(f"dataset = load_from_disk('{args.output_dir}')")
        
        if args.upload:
            print(f"\nDataset uploaded to: https://huggingface.co/datasets/{args.repo_name}")
            print(f"\nTo load from Hub:")
            print(f"from datasets import load_dataset")
            print(f"dataset = load_dataset('{args.repo_name}')")
        
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
