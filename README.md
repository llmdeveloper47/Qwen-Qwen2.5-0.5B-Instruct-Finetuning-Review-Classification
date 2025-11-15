# Qwen/Qwen2.5-0.5B-Instruct-Finetuning-Review-Classification
Repository For Hosting Qwen-3-0.6b Finetuning Code for Amazon 5x5 Review Classification


## Creating Train / Eval / Split
 - Create a dataset folder to save the dataset
 - Refer to the dataset_analysis.ipynb notebook under src/experiment/ to load the dataset and split it into Train / Eval / Test Datasets

## Converting to Huggingface Format and Uploding to Hub
Refer to src/scripts/convert_to_huggingface_dataset.py script to load the locally saved datasets and push to hub ( make sure to create a repository , or just download the dataset from the hub repo provided)

## Finetuning Qwen/Qwen2.5-0.5B-Instruct
 - Create an Account on RunPod
 - Setup a A100 GPU
 - Clone the repo on RunPod  
 - Refer to src/experiments/qwen.ipynb notebook to finetune the model, the best model checkpoin gets saved

##  Pre/Post Training Evaluation on Held Out Test Set
After finetuning is complete, we load the best model checkpoint ( eval loss used as proxy for identifying best model , it could be a custom metric like Recall / F1 too ), refer to the src/experiments/evaluate_models_inference.ipynb notebook to evaluate the pretrained & finetuned model.

## Evaluating GPT on Held Out Test Set
 - Refer to the System Prompt under src/system_prompt/system_prompt_gpt.txt to evaluate GPT 4.1
 - Refer to the src/scripts/gpt_predictions.py to create predictions on the held out test set. Predictions get saved to a folder name.
 - Refer to the src/scripts/evaluate_gpt.py script to evaluate the GPT predictions using the ground truth from the test set.




