import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    # load a model and tokenizer 
    # move the model to the device
    model_name = "my-7b-model" # TODO: change this to a real model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token="HF-TOKEN-IF-NEEDED",
        trust_remote_code=True # IF NEEDED
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token="HF-TOKEN-IF-NEEDED",
        trust_remote_code=True # IF NEEDED
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}")
    return model, tokenizer

def open_knowledge_base(path_to_file: str):
    # open the knowledge base
    # return a string of the knowledge base
    with open(path_to_file, "r", encoding="utf-8") as file:
        return file.read()

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

    # A seperate model call is made to produce the cache
    # An in-scope cache is initialized, passed to the model call, and returned as output

def reset_kv_cache(kv_cache: DynamicCache, input_length: int):
    # Reset the kv cache leaving the processed part intact
    # use DynamicCache.crop(max_length)
    kv_cache.crop(input_length)

def generate(model, tokenizer, user_prompt: str, knowledge_base_kv_cache: DynamicCache, max_new_tokens: int = 300) -> str:
    # Tokenize the user prompt
    input_ids = tokenizer(user_prompt, return_tensors="pt").to(model.device).input_ids
    original_input_length = input_ids.shape[1]
    
    # Track generated tokens
    output_ids = input_ids.clone()
    next_token_input = input_ids
    
    # Manual generation loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Process the next token(s)
            outputs = model(
                input_ids=next_token_input,
                past_key_values=knowledge_base_kv_cache,
                use_cache=True
            )
            
            # Get next token prediction (greedy decoding)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            
            # Update cache for next iteration
            knowledge_base_kv_cache = outputs.past_key_values
            
            # Append new token to output
            output_ids = torch.cat([output_ids, next_token], dim=1)
            
            # For next iteration, only process the new token
            next_token_input = next_token
            
            # Check for EOS token
            if next_token.item() in tokenizer.eos_token_id:
                break
    
    # Extract only the generated part (excluding the input)
    generated_text = tokenizer.decode(output_ids[0, original_input_length:], skip_special_tokens=True)
    return generated_text


def main():
    # TODO: make a chat loop that does the following:
    # 0. out of the loop, produce_kv_cache() once using the original knowledge
    # START LOOP
    # 1. Ask for user prompt
    # 2. generate using the pre-populated kv cache
    # 3. reset_kv_cache() to clean the populated cache to original state
    # 4. End of loop if user prompts "exit"

    # Fill out prompt

    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Load knowledge base
    knowledge_text = open_knowledge_base("./knowledge_base_raw/my_knowledge_base.txt")
    
    # Combine with prompt template
    knowledge_prompt = combine_knowledge_base_and_prompt(knowledge_text)
    
    # Produce initial KV cache
    print("Processing knowledge base...")
    knowledge_cache, knowledge_length = produce_kv_cache(model, tokenizer, knowledge_prompt)
    print("Knowledge base processed and cached.")
    
    # Chat loop
    print("\nChat started. Type 'exit' to end the conversation.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        # Generate response with knowledge cache
        print("\nAssistant: ", end="")
        response = generate(model, tokenizer, user_input, knowledge_cache)
        print(response)
        
        # Reset cache for next question
        knowledge_cache = reset_kv_cache(knowledge_cache, knowledge_length)

if __name__ == "__main__":
    main()
