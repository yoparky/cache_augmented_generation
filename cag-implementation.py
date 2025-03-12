import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig
from typing import Tuple, Optional
import copy
import datetime  # Added for timestamp functionality

def load_model(model_path: str, token: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    try:
        print(f"Loading model from path: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Load model with device_map="auto" but don't call .to(device) afterward
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # This automatically handles device placement
            trust_remote_code=True # Optional: creates a folder for offloaded parameters
        )
        
        if hasattr(model, 'hf_device_map'):
            print(f"Model loaded with device map: {model.hf_device_map}")
        else:
            print("Model loaded with automatic device mapping")
            
        return model, tokenizer
    except ValueError as e:
        raise ValueError(f"Invalid model path or configuration: {str(e)}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            raise RuntimeError("Not enough GPU memory to load the model. Try using quantization: load_in_8bit=True")
        raise RuntimeError(f"Failed to load model: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading model: {str(e)}")

def open_knowledge_base(path_to_file: str):
    # open the knowledge base
    # return a string of the knowledge base
    try:
        with open(path_to_file, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Knowledge base file not found at {path_to_file}")
    except Exception as e:
        raise Exception(f"Error reading knowledge base: {str(e)}")

def combine_knowledge_base_and_prompt(knowledge_base: str) -> str:
    # combine the knowledge base and the prompt
    # return a string of the combined prompt
    return f"""
    <|system|>
    You are an assistant who provides concise factual answers based on the context provided.
    You will only answer the specific question asked, without creating follow-up questions.
    Answer concisely in less than or equal to 5 sentences.
    <|user|>
    Context: 
    {knowledge_base}
    Answer the question:
    """.strip()


def produce_kv_cache(model, tokenizer, combined_prompt) -> tuple[DynamicCache, int]:
    # Preprocess and produce a kv cache with knowledge
    # consider populating a prompt with the knowledge in a nice context prompt
    # and then passing it through the attention layer.
    knowledge_base_kv_cache = DynamicCache()
    # generate the input_ids
    inputs = tokenizer(combined_prompt, return_tensors="pt").to(model.device)
    # since kv cache is 1:1 input-bound, we store the number of input ids
    input_length = inputs.input_ids.shape[1]

    # populate the knowledge base kv cache
    # one-pass to populate the cache
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=knowledge_base_kv_cache,
            use_cache=True
        )
    
    # Return populated cache and the original sequence length
    return outputs.past_key_values, input_length

def reset_kv_cache(kv_cache: DynamicCache, input_length: int) -> None:
    # Reset the kv cache leaving the processed part intact
    kv_cache.crop(input_length)

def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    user_prompt: str,
    knowledge_base_kv_cache: DynamicCache,
    max_new_tokens: int = 50,
    temperature: float = 1.0
) -> str:
    try:
        # Tokenize the user prompt
        input_ids = tokenizer(user_prompt, return_tensors="pt").to(model.device).input_ids
        original_input_length = input_ids.shape[1]
        
        # Track generated tokens
        output_ids = input_ids.clone()
        next_token_input = input_ids
        
        # Manual generation loop
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=next_token_input,
                    past_key_values=knowledge_base_kv_cache,
                    use_cache=True
                )
                
                # Apply temperature to logits
                next_token_logits = outputs.logits[:, -1, :] / temperature
                next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
                
                # Update the cache in-place
                knowledge_base_kv_cache = outputs.past_key_values
                output_ids = torch.cat([output_ids, next_token], dim=1)
                next_token_input = next_token
                
                if model.config.eos_token_id is not None and next_token.item() == model.config.eos_token_id:
                    break
        
        return tokenizer.decode(output_ids[0, original_input_length:], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")

def get_timestamp() -> datetime.datetime:
    """Get the current timestamp as a datetime object."""
    return datetime.datetime.now()

def format_timestamp(timestamp: datetime.datetime) -> str:
    """Format a datetime object as a readable string with millisecond precision."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def calculate_elapsed_time(start_time: datetime.datetime, end_time: datetime.datetime) -> float:
    """Calculate elapsed time between two timestamps in seconds with millisecond precision."""
    delta = end_time - start_time
    return delta.total_seconds()

def main():
    try:
        # Load model and tokenizer
        model, tokenizer = load_model("./Qwen2.5-1.5B-Instruct")
        
        # Load knowledge base
        knowledge_text = open_knowledge_base("./knowledge_base_raw/my_knowledge_base.txt")
        
        # Combine with prompt template
        knowledge_prompt = combine_knowledge_base_and_prompt(knowledge_text)
        
        # Produce initial KV cache
        print("Processing knowledge base...")
        knowledge_base_kv_cache, knowledge_base_input_length = produce_kv_cache(model, tokenizer, knowledge_prompt)
        print("Knowledge base processed and cached.")
        
        
        # Chat loop
        print("\nChat started. Type 'exit' to end the conversation.")
        while True:
            try:
                user_input = input("\nYou: ")
                # Capture input timestamp as datetime object
                input_timestamp = get_timestamp()
                print(f"[Input received at: {format_timestamp(input_timestamp)}]")
                
                if user_input.lower() == "exit":
                    break
                
                print("\nAssistant: ", end="")
                response = generate(model, tokenizer, user_input, knowledge_base_kv_cache)
                print(response)
                
                # Capture output timestamp as datetime object
                output_timestamp = get_timestamp()
                print(f"[Response completed at: {format_timestamp(output_timestamp)}]")
                
                # Calculate and display elapsed time
                elapsed_seconds = calculate_elapsed_time(input_timestamp, output_timestamp)
                print(f"[CAG Response time: {elapsed_seconds:.3f} seconds]")
                
                # Reset cache for next question
                reset_kv_cache(knowledge_base_kv_cache, knowledge_base_input_length)
            except KeyboardInterrupt:
                print("\nChat interrupted by user.")
                break
            except Exception as e:
                print(f"\nError during generation: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return

if __name__ == "__main__":
    main()