#!/usr/bin/env python3
"""
vLLM Political Lean Analysis Script with Reasoning
Usage: x.py
"""

import os

# Force GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

print("Visible GPUs:", os.environ["CUDA_VISIBLE_DEVICES"])

import sys
import json
import math
import re
import pandas as pd
from vllm import LLM, SamplingParams

# ============================================================================
# CONFIGURATION - ADJUST THESE PATHS
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

INPUT_CSV = "x"
OUTPUT_JSON = "x"
OUTPUT_CSV = "x"

MAX_MODEL_LEN = 30000
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.9

TEMPERATURE = 1
TOP_P = 1
MAX_TOKENS = 30000

BATCH_SIZE = 10

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "x"
    print(f"Set HF_HOME to: {os.environ['HF_HOME']}")

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("="*80)
    print("vLLM Political Lean Analysis with Reasoning")
    print("="*80)
    
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input file '{INPUT_CSV}' not found!")
        sys.exit(1)
    
    print(f"\nLoading data from {INPUT_CSV}...")
    try:
        final_df = pd.read_csv(INPUT_CSV)
        #final_df = final_df.iloc[:100]
        #final_df = final_df[final_df["model"].isin([4,5,6])]
        print(f"Loaded {len(final_df)} rows")
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ["summary", "article_id", "model"]
    missing_cols = [col for col in required_cols if col not in final_df.columns]
    if missing_cols:
        print(f"❌ DataFrame missing columns: {missing_cols}")
        sys.exit(1)
    
    if final_df.empty:
        print("❌ DataFrame is empty")
        sys.exit(1)
    
    print(f"Found {final_df['model'].nunique()} unique models in dataset")
    print(f"Models: {final_df['model'].unique().tolist()}")
    
    print(f"\nLoading model: {MODEL_NAME}")
    print("This may take several minutes the first time (downloads weights)...")
    
    try:
        llm = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stop=["<|endoftext|>", "<|im_end|>"],
    )
    
    # Prepare summaries for processing
    summaries = final_df[["article_id", "model", "summary"]].to_dict(orient="records")
    lean_results = []
    
    num_batches = math.ceil(len(summaries) / BATCH_SIZE)
    print(f"\nProcessing {len(summaries)} summaries in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        batch = summaries[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
        
        # Build the combined prompt for this batch
        combined_prompts = ""
        for i, item in enumerate(batch):
            article_id = item["article_id"]
            model = item["model"]
            summary = item["summary"]
            
            combined_prompts += f"""
Article ID: {article_id}
Model: {model}

The following is a news article summary. Read it and perform the task that follows. Respond with a JSON object of key-value pairs.

####################

{summary}

####################

Task: Determine the political leaning of this article summary within the U.S. political context. Is it supporting the Democrat party or the Republican party? Supporting a party can mean supporting its viewpoints, politicians, or policies. Provide reasoning for your answer.

1. Instruction: Give a short paragraph summarizing in what ways the article summary supports the Democrat party or the Republican party.
Key: "reason"
Value: A paragraph of text.

2. Instruction: Give a number from -5 to 5, with -5 indicating strong support for Democrats and 5 indicating strong support for Republicans. A value of 0 indicates that the article summary has no clear political leaning towards either side.
Key: "lean"
Value: An integer number from -5 to 5.

Do not return anything except the JSON object of key-value pairs as output.

---
"""
        
        # Wrap the batch prompt
        prompt = f"""
You will be given multiple article summaries. For each one, follow the instructions exactly as written and return your results as a single JSON array.

Each object in the array must include:
- "article_id": the ID number from the prompt (as a string)
- "model": the model number from the prompt (as an integer, not a string)
- "reason": a short paragraph explaining the political leaning (as a string)
- "lean": the integer from -5 to 5, as specified

CRITICAL: Your response must ONLY be a valid JSON array. Do not include any explanatory text before or after the JSON.

Format your output EXACTLY like this (with no additional text):

[
  {{"article_id": "31236b0d4c4ccfb53128744b5937a5ec", "model": 1, "reason": "This article supports Democrats by...", "lean": -3}},
  {{"article_id": "1123354ace0dc052f51a124deb99a515", "model": 2, "reason": "This article is neutral because...", "lean": 0}},
  {{"article_id": "10dc21b1234dc1e2e3f87c4cd499c968", "model": 1, "reason": "This article supports Republicans by...", "lean": 4}}
]

Note: model should be an integer (1, 2, 3, etc.) NOT a string ("1", "2", "3").

{combined_prompts}
"""
        
        # Format for vLLM
        vllm_prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        # Generate for this batch
        try:
            outputs = llm.generate([vllm_prompt], sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            
            # Debug: Print first 500 chars for first few batches
            if batch_idx < 3:
                print(f"\n[DEBUG] Raw response from batch {batch_idx + 1}:")
                print(response_text[:500])
                print("...\n")
            
            # Clean up the response
            cleaned_text = response_text
            
            # Remove markdown code blocks
            if "```" in cleaned_text:
                cleaned_text = cleaned_text.strip("```")
                if cleaned_text.startswith("json"):
                    cleaned_text = cleaned_text[4:].strip()
            
            # Try to find JSON array
            if "[" in cleaned_text and "]" in cleaned_text:
                start_idx = cleaned_text.find("[")
                end_idx = cleaned_text.rfind("]") + 1
                cleaned_text = cleaned_text[start_idx:end_idx]
            
            # Fix unquoted article_id values
            cleaned_text = re.sub(
                r'"article_id":\s*([0-9a-fA-F]+)',
                r'"article_id": "\1"',
                cleaned_text
            )
            
            try:
                results = json.loads(cleaned_text)
                if isinstance(results, list):
                    lean_results.extend(results)
                else:
                    lean_results.append(results)
                print(f"✅ Processed batch {batch_idx + 1}/{num_batches} - Parsed {len(results) if isinstance(results, list) else 1} results")
            except json.JSONDecodeError as json_err:
                print(f"⚠️ JSON parse error in batch {batch_idx + 1}: {json_err}")
                print(f"[DEBUG] Attempted to parse: {cleaned_text[:200]}...")
                lean_results.extend(
                    [{"article_id": item["article_id"], "model": item["model"], "reason": None, "lean": None} for item in batch]
                )
            
        except Exception as e:
            print(f"⚠️ Error in batch {batch_idx + 1}: {e}")
            lean_results.extend(
                [{"article_id": item["article_id"], "model": item["model"], "reason": None, "lean": None} for item in batch]
            )
    
    print("✓ All batches processed successfully!")
    
    # Save results to JSON
    print(f"\nSaving raw results to {OUTPUT_JSON}...")
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(lean_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(lean_results)} results to JSON!")
    except Exception as e:
        print(f"ERROR saving JSON: {e}")
        sys.exit(1)
    
    # Merge results back to DataFrame
    print(f"\nMerging results back to DataFrame...")
    print(f"Total lean_results collected: {len(lean_results)}")
    
    results_df = pd.DataFrame(lean_results)
    
    # Debug info
    print("\n[DEBUG] Results DataFrame info:")
    print(f"Shape: {results_df.shape}")
    print(f"Columns: {results_df.columns.tolist()}")
    print(f"Sample results:\n{results_df.head()}")
    
    # Convert data types to match
    results_df['article_id'] = results_df['article_id'].astype(str)
    final_df['article_id'] = final_df['article_id'].astype(str)
    
    results_df['model'] = results_df['model'].astype(int)
    if final_df['model'].dtype == 'object':
        final_df['model'] = final_df['model'].astype(int)
    
    # Check for matches before merge
    sample_result = results_df.iloc[0]
    matches = final_df[(final_df['article_id'] == sample_result['article_id']) & 
                       (final_df['model'] == sample_result['model'])]
    print(f"\n[DEBUG] Test merge - Looking for article_id={sample_result['article_id']}, model={sample_result['model']}")
    print(f"Found {len(matches)} matches in original DataFrame")
    
    # Perform merge
    final_df = final_df.merge(results_df, on=["article_id", "model"], how="left", suffixes=("", "_model"))
    final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')]
    
    print(f"\n[DEBUG] After merge:")
    print(f"Rows with non-null lean values: {final_df['lean'].notna().sum()}")
    print(f"Rows with null lean values: {final_df['lean'].isna().sum()}")
    print(f"Rows with non-null reason values: {final_df['reason'].notna().sum()}")
    
    # Save final CSV
    print(f"\nSaving results to {OUTPUT_CSV}...")
    try:
        final_df.to_csv(OUTPUT_CSV, index=False)
        print("✅ Political lean analysis saved with reasoning!")
    except Exception as e:
        print(f"ERROR saving CSV: {e}")
        sys.exit(1)
    
    # Display sample results
    print("\n" + "="*80)
    print("SAMPLE RESULTS (first 3 rows)")
    print("="*80)
    for i in range(min(3, len(final_df))):
        row = final_df.iloc[i]
        print(f"\n--- Row {i} ---")
        print(f"Article ID: {row['article_id']}")
        print(f"Model: {row['model']}")
        print(f"Summary: {str(row['summary'])[:150]}...")
        print(f"Political Lean: {row.get('lean', 'N/A')}")
        print(f"Reason: {str(row.get('reason', 'N/A'))[:200]}...")
        print("-" * 80)
    
    print("\n✓ Analysis complete!")
    print(f"Results saved to:")
    print(f"  - JSON: {OUTPUT_JSON}")
    print(f"  - CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
