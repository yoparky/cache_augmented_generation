import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional

def load_model(model_name: str = "my-7b-model", token: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            token=token,
            trust_remote_code=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Model loaded on {device}")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

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
    prompt = """

    """
    return f"{prompt}\n\n{knowledge_base}".strip()


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
    kv_cache.crop(end=input_length)

def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    knowledge_base_kv_cache: DynamicCache,
    max_new_tokens: int = 300,
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
                
                knowledge_base_kv_cache = outputs.past_key_values
                output_ids = torch.cat([output_ids, next_token], dim=1)
                next_token_input = next_token
                
                if next_token.item() in tokenizer.eos_token_id:
                    break
        
        return tokenizer.decode(output_ids[0, original_input_length:], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")

def main():
    try:
        # Load model and tokenizer
        model, tokenizer = load_model()
        
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
                if user_input.lower() == "exit":
                    break
                
                print("\nAssistant: ", end="")
                response = generate(model, tokenizer, user_input, knowledge_base_kv_cache)
                print(response)
                
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
