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

    AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token="HF-TOKEN-IF-NEEDED",
        trust_remote_code=True # IF NEEDED
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}")

def open_knowledge_base(path_to_file: str):
    # open the knowledge base
    # return a string of the knowledge base
    with open(path_to_file, "r", encoding="utf-8") as file:
        return file.read()

def combine_knowledge_base_and_prompt(knowledge_base: str):
    # combine the knowledge base and the prompt
    # return a string of the combined prompt
    prompt = """

    """
    return f"{prompt}\n\n{knowledge_base}".strip()


def produce_kv_cache(model, tokenizer, combined_prompt) -> DynamicCache, int:
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
    pass

def reset_kv_cache() -> DynamicCache:
    # Reset the kv cache leaving the processed part intact
    # use DynamicCache.crop(max_length)
    pass

def generate():
    # customized decoder model call using the generated DynamicCache passed in as
    # the past_key_values argument

    pass



# TODO: make a chat loop that does the following:
# 0. out of the loop, produce_kv_cache() once using the original knowledge
# START LOOP
# 1. Ask for user prompt
# 2. generate using the pre-populated kv cache
# 3. reset_kv_cache() to clean the populated cache to original state
# 4. End of loop if user prompts "exit"


