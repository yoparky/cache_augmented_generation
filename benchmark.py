import time
import json
import pandas as pd
import CAG
import RAG

def benchmark_models(model_id, embedding_model_id, initial_text, prompts, max_new_tokens=100):
    """
    Run timing benchmarks on both CAG and RAG models with the same inputs.
    
    Args:
        model_id: HuggingFace model ID
        embedding_model_id: HuggingFace embedding model ID (for RAG)
        initial_text: Text to populate the model context
        prompts: List of prompts to test
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Dictionary with timing results
    """
    results = {
        "model_id": model_id,
        "embedding_model_id": embedding_model_id,
        "max_new_tokens": max_new_tokens,
        "cag_results": [],
        "rag_results": []
    }
    
    # Initialize tokenizers once to count tokens
    cag_tokenizer = None
    rag_embedding_tokenizer = None
    
    # Benchmark CAG
    print(f"Running CAG benchmark with {len(prompts)} prompts...")
    with CAG.CAG(model_id, initial_text, max_new_tokens=max_new_tokens) as cag:
        cag_tokenizer = cag.tokenizer
        for i, prompt in enumerate(prompts):
            # Count tokens for the full input (initial text + prompt)
            full_input = initial_text + prompt
            input_tokens = len(cag.tokenizer.encode(full_input))
            
            start_time = time.time()
            response = cag.cag_decode([prompt])[0]
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            results["cag_results"].append({
                "prompt": prompt,
                "input_tokens": input_tokens,
                "execution_time": execution_time,
                "response": response
            })
            
            print(f"CAG Prompt {i+1}/{len(prompts)}: {execution_time:.2f} seconds, {input_tokens} tokens")
    
    # Benchmark RAG
    print(f"\nRunning RAG benchmark with {len(prompts)} prompts...")
    with RAG.RAG(model_id, embedding_model_id, initial_text, max_new_tokens=max_new_tokens) as rag:
        rag_embedding_tokenizer = rag.embedding_tokenizer
        for i, prompt in enumerate(prompts):
            # Get the prompt tokens
            prompt_tokens = len(rag.tokenizer.encode(prompt))
            
            # First retrieve the context to measure total tokens
            context = rag.retrieve_context(prompt)
            
            # Measure the formatted prompt that includes context
            formatted_prompt = f"""Based on the following context, {prompt}

Context:
{context}

Answer:"""
            total_tokens = len(rag.tokenizer.encode(formatted_prompt))
            
            # Now time the actual generation
            start_time = time.time()
            response = rag.cag_decode([prompt])[0]
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Also count tokens for the embedding model
            embedding_tokens = len(rag.embedding_tokenizer.encode(prompt))
            
            results["rag_results"].append({
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,  # This includes prompt + context
                "embedding_tokens": embedding_tokens,
                "execution_time": execution_time,
                "response": response
            })
            
            print(f"RAG Prompt {i+1}/{len(prompts)}: {execution_time:.2f} seconds, {prompt_tokens} prompt tokens, {total_tokens} total tokens (with context), {embedding_tokens} embedding tokens")
    
    return results

def create_comparison_table(results):
    """
    Create a pandas DataFrame comparing CAG and RAG performance.
    
    Args:
        results: Dictionary with benchmark results
        
    Returns:
        DataFrame with timing comparison
    """
    table_data = []
    
    for i in range(len(results["cag_results"])):
        cag_result = results["cag_results"][i]
        rag_result = results["rag_results"][i]
        
        # Ensure we're comparing the same prompt
        assert cag_result["prompt"] == rag_result["prompt"], "Prompt mismatch in results"
        
        table_data.append({
            "prompt": cag_result["prompt"],
            "cag_tokens": cag_result.get("input_tokens", 0),
            "rag_prompt_tokens": rag_result.get("prompt_tokens", 0),
            "rag_total_tokens": rag_result.get("total_tokens", 0),
            "cag_time": cag_result["execution_time"],
            "rag_time": rag_result["execution_time"],
            "time_difference": rag_result["execution_time"] - cag_result["execution_time"],
            "time_ratio": rag_result["execution_time"] / cag_result["execution_time"] if cag_result["execution_time"] > 0 else float('inf'),
            "token_ratio": rag_result.get("total_tokens", 0) / cag_result.get("input_tokens", 1) if cag_result.get("input_tokens", 0) > 0 else float('inf')
        })
    
    df = pd.DataFrame(table_data)
    
    # Add summary row with averages
    summary = pd.DataFrame([{
        "prompt": "AVERAGE",
        "cag_tokens": df["cag_tokens"].mean(),
        "rag_prompt_tokens": df["rag_prompt_tokens"].mean(),
        "rag_total_tokens": df["rag_total_tokens"].mean(),
        "cag_time": df["cag_time"].mean(),
        "rag_time": df["rag_time"].mean(),
        "time_difference": df["time_difference"].mean(),
        "time_ratio": df["time_ratio"].mean(),
        "token_ratio": df["token_ratio"].mean()
    }])
    
    # Use pd.concat instead of append (which is deprecated)
    return pd.concat([df, summary], ignore_index=True)

def save_results(results, output_file="benchmark_results.json"):
    """Save benchmark results to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

def save_table(df, output_file="comparison_table.csv"):
    """Save comparison table to a CSV file"""
    df.to_csv(output_file, index=False)
    print(f"Comparison table saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Load initial text from the file used in both implementations
    with open("initial_text.txt", "r") as f:
        initial_text = f.read()
    
    # Test prompts
    prompts = [
        "What happens in the prologue of A Game of Thrones?",
        "Who is Ser Waymar Royce?",
        "Describe the relationship between Will and Gared.",
        "What is the Night's Watch?",
        "Why does Gared want to turn back?"
    ]
    
    # Run the benchmark
    print("Starting benchmark...")
    results = benchmark_models(model_id, embedding_model_id, initial_text, prompts)
    
    # Save raw results
    save_results(results)
    
    # Create and save comparison table
    table = create_comparison_table(results)
    save_table(table)
    
    # Print comparison table
    print("\nComparison Table:")
    print(table.to_string())
