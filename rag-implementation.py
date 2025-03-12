import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import Tuple, Optional, List
import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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
            trust_remote_code=True
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

def open_knowledge_base(path_to_file: str) -> str:
    """Open the knowledge base file and return its contents as a string."""
    try:
        with open(path_to_file, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Knowledge base file not found at {path_to_file}")
    except Exception as e:
        raise Exception(f"Error reading knowledge base: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks of approximately chunk_size characters.
    
    Args:
        text: The text to split
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        # Find the end of the chunk
        end = min(start + chunk_size, text_len)
        
        # If not at the end of the text, try to find a good breaking point
        if end < text_len:
            # Try to find a newline or period to break at
            newline_pos = text.rfind('\n', start, end)
            period_pos = text.rfind('. ', start, end)
            space_pos = text.rfind(' ', start, end)
            
            # Use the latest good breaking point
            if newline_pos > start:
                end = newline_pos + 1
            elif period_pos > start:
                end = period_pos + 2
            elif space_pos > start:
                end = space_pos + 1
        
        # Add the chunk to the list
        chunks.append(text[start:end])
        
        # Move the start position, considering overlap
        start = end - overlap if end < text_len else text_len
    
    return chunks

class RAGSystem:
    def __init__(self, knowledge_base_path: str, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG system with a knowledge base.
        
        Args:
            knowledge_base_path: Path to the knowledge base text file
            embedding_model_name: Name of the SentenceTransformer model to use
        """
        print(f"Initializing RAG system with {knowledge_base_path}...")
        
        # Load the knowledge base
        self.knowledge_text = open_knowledge_base(knowledge_base_path)
        
        # Chunk the knowledge base
        self.chunks = chunk_text(self.knowledge_text)
        print(f"Knowledge base chunked into {len(self.chunks)} segments")
        
        # Initialize the embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Create embeddings for the chunks
        print("Creating embeddings for knowledge chunks...")
        self.embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        
        # Build the Faiss index
        print("Building Faiss index...")
        self.index = self._build_faiss_index(self.embeddings)
        
        print(f"RAG system initialized successfully with {len(self.chunks)} chunks")

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build a Faiss index from embeddings."""
        # Get dimensions of the embeddings
        d = embeddings.shape[1]
        
        # Create a Faiss index - using L2 distance (Euclidean)
        index = faiss.IndexFlatL2(d)
        
        # Add embeddings to the index
        index.add(embeddings)
        
        return index

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of top results to return
        
        Returns:
            Combined text of the top matching chunks
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode([query])
        
        # Search the index
        D, I = self.index.search(query_embedding, top_k)
        
        # Collect the retrieved chunks
        retrieved_chunks = [self.chunks[i] for i in I[0]]
        
        # Combine the chunks into a single text
        return "\n\n".join(retrieved_chunks)

def combine_knowledge_and_prompt(query: str, retrieved_context: str) -> str:
    """
    Combine retrieved context and user query into a prompt.
    
    Args:
        query: User query
        retrieved_context: Retrieved context from the RAG system
    
    Returns:
        Combined prompt
    """
    return f"""
    <|system|>
    You are an assistant who provides concise factual answers based on the context provided.
    You will only answer the specific question asked, without creating follow-up questions.
    Answer concisely in less than or equal to 5 sentences.
    <|user|>
    Context: 
    {retrieved_context}
    
    Answer the question: {query}
    """.strip()

def produce_kv_cache(model, tokenizer, combined_prompt) -> tuple[DynamicCache, int]:
    """Create a KV cache from the prompt."""
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

def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    user_prompt: str,
    knowledge_base_kv_cache: DynamicCache,
    max_new_tokens: int = 50,
    temperature: float = 1.0
) -> str:
    try:
        # For RAG implementation, user_prompt might be empty as we've already 
        # included it in the context. Use a token to start generation if needed.
        if user_prompt:
            input_ids = tokenizer(user_prompt, return_tensors="pt").to(model.device).input_ids
        else:
            # If no prompt is provided, use a suitable token to start generation
            input_ids = torch.tensor([[tokenizer.eos_token_id]]).to(model.device)
        
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
        
        # Initialize the RAG system
        rag_system = RAGSystem("./knowledge_base_raw/my_knowledge_base.txt")
        
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
                
                # Retrieve relevant context
                print("[Retrieving relevant information...]")
                retrieved_context = rag_system.retrieve_relevant_chunks(user_input)
                
                # Combine query and context
                rag_prompt = combine_knowledge_and_prompt(user_input, retrieved_context)
                
                # Create a fresh KV cache for this prompt
                rag_kv_cache = DynamicCache()
                
                print("\nAssistant: ", end="")
                response = generate(model, tokenizer, rag_prompt, rag_kv_cache)
                print(response)
                
                # Capture output timestamp as datetime object
                output_timestamp = get_timestamp()
                print(f"[Response completed at: {format_timestamp(output_timestamp)}]")
                
                # Calculate and display elapsed time
                elapsed_seconds = calculate_elapsed_time(input_timestamp, output_timestamp)
                print(f"[RAG Response time: {elapsed_seconds:.3f} seconds]")
                
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