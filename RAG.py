import copy
import torch
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from accelerate.test_utils.testing import get_backend
from huggingface_hub import login
from typing import List, Tuple
import textwrap

login(token = "hf_???")

class RAG:
    def __init__(self, model_id: str, embedding_model_id: str, initial_text: str, max_new_tokens: int = 20, 
                chunk_size: int = 200, chunk_overlap: int = 50, top_k: int = 3):
        self.device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_new_tokens = max_new_tokens
        
        # Set up embedding model for RAG
        self.embedding_model = AutoModel.from_pretrained(embedding_model_id).to(self.device)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
        
        # RAG parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize FAISS index and document store
        self.setup_rag(initial_text)
    
    def __eq__(self, other):
        return (self.model == other.model and 
                self.tokenizer == other.tokenizer and 
                self.embedding_model == other.embedding_model)
    
    def __hash__(self):
        return hash((self.model, self.tokenizer, self.embedding_model))
    
    def close(self):
        """Explicitly clean up resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'embedding_model'):
            del self.embedding_model
        torch.cuda.empty_cache()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        self.close()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for a given text, handling long sequences."""
        # Check if text is too long
        tokens = self.embedding_tokenizer.encode(text)
        max_length = self.embedding_tokenizer.model_max_length
        
        if len(tokens) <= max_length:
            # If text fits within the model's max length, process normally
            inputs = self.embedding_tokenizer(text, return_tensors="pt", 
                                            padding=True, truncation=True, 
                                            max_length=max_length).to(self.device)
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            
            # Use mean pooling for the embedding
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings
        else:
            # For longer texts, split into chunks and average the embeddings
            embeddings_list = []
            
            # Split into chunks that fit within max_length
            for i in range(0, len(tokens), max_length - 100):  # -100 for safety margin
                chunk_tokens = tokens[i:i + max_length - 100]
                chunk_text = self.embedding_tokenizer.decode(chunk_tokens)
                
                chunk_inputs = self.embedding_tokenizer(chunk_text, return_tensors="pt", 
                                                      padding=True, truncation=True, 
                                                      max_length=max_length).to(self.device)
                
                with torch.no_grad():
                    chunk_outputs = self.embedding_model(**chunk_inputs)
                
                chunk_embedding = chunk_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings_list.append(chunk_embedding)
            
            # Average all chunk embeddings
            final_embedding = np.mean(embeddings_list, axis=0)
            return final_embedding
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        tokens = self.embedding_tokenizer.encode(text)
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.embedding_tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # If we've reached the end of the text, break
            if i + self.chunk_size >= len(tokens):
                break
        
        return chunks
    
    def setup_rag(self, text: str):
        """Set up FAISS index with the provided text."""
        # Chunk the text
        self.chunks = self.chunk_text(text)
        
        # Create embeddings for each chunk
        embeddings = []
        for chunk in self.chunks:
            embedding = self.get_embedding(chunk)
            embeddings.append(embedding)
        
        # Convert to numpy array and ensure correct shape
        embeddings_array = np.vstack(embeddings)
        
        # Create FAISS index (L2 distance)
        self.embedding_dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        
        # Add embeddings to the index
        self.index.add(embeddings_array)
        
        print(f"RAG system initialized with {len(self.chunks)} chunks")
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context based on the query."""
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Search for similar chunks
        distances, indices = self.index.search(query_embedding, self.top_k)
        
        # Retrieve the top k chunks
        retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
        
        # Join the chunks to create the context
        context = "\n\n".join(retrieved_chunks)
        
        return context
    
    def cag_decode_batch(self, prompts: list[str]):
        if not prompts: return []
        
        responses = []
        for prompt in prompts:
            try:
                # Retrieve relevant context for this prompt
                context = self.retrieve_context(prompt)
                
                # Format the prompt with the retrieved context
                formatted_prompt = f"""Based on the following context, {prompt}

Context:
{context}

Answer:"""
                
                # Tokenize the prompt
                new_inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                
                # Generate the response
                outputs = self.model.generate(
                    **new_inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True, 
                    output_scores=False
                )
                
                # Decode the generated tokens
                response = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                responses.append(response)
            except Exception as e:
                print(f"Error decoding prompt: {e}")
                responses.append(None)
        return responses
    
    def cag_decode(self, prompts: list[str]):
        if not prompts: return []
        
        responses = []
        for prompt in prompts:
            try:
                # Retrieve relevant context for this prompt
                context = self.retrieve_context(prompt)
                
                # Format the prompt with the retrieved context
                formatted_prompt = f"""Based on the following context, {prompt}

Context:
{context}

Answer:"""
                
                # Tokenize the prompt
                new_inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                
                # Generate the response
                outputs = self.model.generate(
                    **new_inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True, 
                    output_scores=False,
                    # generation control
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.1,
                )
                
                # Decode only the model's answer (not the prompt + context)
                full_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                
                # Try to extract just the answer part
                answer_parts = full_output.split("Answer:")
                if len(answer_parts) > 1:
                    response = answer_parts[1].strip()
                else:
                    response = full_output.replace(formatted_prompt, "").strip()
                
                responses.append(response)
            except Exception as e:
                print(f"Error decoding prompt: {e}")
                responses.append(None)
        return responses


def main():
    """Example usage of the RAG_CAG class."""
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller, efficient embedding model
    
    initial_text = """
    PROLOGUE, A Game of Thrones
    We should start back," Gared urged as the woods began to grow dark around them. "The
    wildlings are dead."
    "Do the dead frighten you?" Ser Waymar Royce asked with just the hint of a smile.
    Gared did not rise to the bait. He was an old man, past fifty, and he had seen the
    lordlings come and go. "Dead is dead," he said. "We have no business with the dead."
    "Are they dead?" Royce asked softly. "What proof have we?"
    "Will saw them," Gared said. "If he says they are dead, that's proof enough for me."
    Will had known they would drag him into the quarrel sooner or later. He wished it had
    been later rather than sooner. "My mother told me that dead men sing no songs," he put
    in.
    "My wet nurse said the same thing, Will," Royce replied. "Never believe anything you
    hear at a woman's tit. There are things to be learned even from the dead." His voice
    echoed, too loud in the twilit forest.
    "We have a long ride before us," Gared pointed out. "Eight days, maybe nine. And night
    is falling."
    Ser Waymar Royce glanced at the sky with disinterest. "It does that every day about this
    time. Are you unmanned by the dark, Gared?"
    Will could see the tightness around Gared's mouth, the barely suppressed anger in his
    eyes under the thick black hood of his cloak. Gared had spent forty years in the Night's
    Watch, man and boy, and he was not accustomed to being made light of. Yet it was more
    than that. Under the wounded pride, Will could sense something else in the older man.
    You could taste it; a nervous tension that came perilous close to fear.
    Will shared his unease. He had been four years on the Wall. The first time he had been
    sent beyond, all the old stories had come rushing back, and his bowels had turned to
    water. He had laughed about it afterward. He was a veteran of a hundred rangings by
    now, and the endless dark wilderness that the southron called the haunted forest had no
    more terrors for him.
    Until tonight. Something was different tonight. There was an edge to this darkness that
    made his hackles rise. Nine days they had been riding, north and northwest and then
    north again, farther and farther from the Wall, hard on the track of a band of wildling
    raiders. Each day had been worse than the day that had come before it. Today was the
    worst of all. A cold wind was blowing out of the north, and it made the trees rustle like
    living things. All day, Will had felt as though something were watching him, something
    cold and implacable that loved him not. Gared had felt it too. Will wanted nothing so
    much as to ride hellbent for the safety of the Wall, but that was not a feeling to share
    with your commander.
    Especially not a commander like this one.
    Ser Waymar Royce was the youngest son of an ancient house with too many heirs. He
    was a handsome youth of eighteen, grey-eyed and graceful and slender as a knife.
    Mounted on his huge black destrier, the knight towered above Will and Gared on their
    smaller garrons. He wore black leather boots, black woolen pants, black moleskin gloves,
    and a fine supple coat of gleaming black ringmail over layers of black wool and boiled
    leather. Ser Waymar had been a Sworn Brother of the Night's Watch for less than half a
    year, but no one could say he had not prepared for his vocation. At least insofar as his
    wardrobe was concerned.
    His cloak was his crowning glory; sable, thick and black and soft as sin. "Bet he killed
    them all himself, he did," Gared told the barracks over wine, "twisted their little heads
    off, our mighty warrior." They had all shared the laugh.
    It is hard to take orders from a man you laughed at in your cups, Will reflected as he sat
    shivering atop his garron. Gared must have felt the same.
    "Mormont said as we should track them, and we did," Gared said. "They're dead. They
    shan't trouble us no more. There's hard riding before us. I don't like this weather. If it
    snows, we could be a fortnight getting back, and snow's the best we can hope for. Ever
    seen an ice storm, my lord?"
    The lordling seemed not to hear him. He studied the deepening twilight in that half-
    bored, half-distracted way he had. Will had ridden with the knight long enough to
    understand that it was best not to interrupt him when he looked like that. "Tell me again
    what you saw, Will. All the details. Leave nothing out."
    Will had been a hunter before he joined the Night's Watch. Well, a poacher in truth.
    Mallister freeriders had caught him red-handed in the Mallisters' own woods, skinning
    one of the Mallisters' own bucks, and it had been a choice of putting on the black or
    losing a hand. No one could move through the woods as silent as Will, and it had not
    taken the black brothers long to discover his talent.
    "The camp is two miles farther on, over that ridge, hard beside a stream," Will said. "I
    got close as I dared. There's eight of them, men and women both. No children I could
    see. They put up a lean-to against the rock. The snow's pretty well covered it now, but I
    could still make it out. No fire burning, but the firepit was still plain as day. No one
    moving. I watched a long time. No living man ever lay so still."
    "Did you see any blood?"
    "Well, no," Will admitted.
    "Did you see any weapons?"
    "Some swords, a few bows. One man had an axe. Heavy-looking, double-bladed, a cruel
    piece of iron. It was on the ground beside him, right by his hand."
    "Did you make note of the position of the bodies?"
    Will shrugged. "A couple are sitting up against the rock. Most of them on the ground.
    Fallen, like."
    "Or sleeping," Royce suggested.
    "Fallen," Will insisted. "There's one woman up an ironwood, half-hid in the branches. A
    far-eyes." He smiled thinly. "I took care she never saw me. When I got closer, I saw that
    she wasn't moving neither." Despite himself, he shivered.
    "You have a chill?" Royce asked.
    "Some," Will muttered. "The wind, m'lord."
    The young knight turned back to his grizzled man-at-arms. Frostfallen leaves whispered
    past them, and Royce's destrier moved restlessly. "What do you think might have killed
    these men, Gared?" Ser Waymar asked casually. He adjusted the drape of his long sable
    cloak.
    "It was the cold," Gared said with iron certainty. "I saw men freeze last winter, and the
    one before, when I was half a boy. Everyone talks about snows forty foot deep, and how
    the ice wind comes howling out of the north, but the real enemy is the cold. It steals up
    on you quieter than Will, and at first you shiver and your teeth chatter and you stamp
    your feet and dream of mulled wine and nice hot fires. It burns, it does. Nothing burns
    like the cold. But only for a while. Then it gets inside you and starts to fill you up, and
    after a while you don't have the strength to fight it. It's easier just to sit down or go to
    sleep. They say you don't feel any pain toward the end. First you go weak and drowsy,
    and everything starts to fade, and then it's like sinking into a sea of warm milk. Peaceful,
    like."
    "Such eloquence, Gared," Ser Waymar observed. "I never suspected you had it in you."
    "I've had the cold in me too, lordling." Gared pulled back his hood, giving Ser Waymar a
    good long look at the stumps where his ears had been. "Two ears, three toes, and the
    little finger off my left hand. I got off light. We found my brother frozen at his watch,
    with a smile on his face."
    Ser Waymar shrugged. "You ought dress more warmly, Gared."
    Gared glared at the lordling, the scars around his ear holes flushed red with anger where
    Maester Aemon had cut the ears away. "We'll see how warm you can dress when the
    winter comes." He pulled up his hood and hunched over his garron, silent and sullen.
    "If Gared said it was the cold . . . " Will began.
    """

    try:
        # Use context manager for automatic cleanup
        with RAG(model_id, embedding_model_id, initial_text, max_new_tokens=100) as rag_cag:
            prompts = [
                "What happens in the prologue of A Game of Thrones?",
                "Who is Ser Waymar?"
            ]
            
            responses = rag_cag.cag_decode(prompts)
            
            for prompt, response in zip(prompts, responses):
                print(f"Q: {prompt}")
                print(f"A: {response}")
                print("-" * 50)
    except Exception as e:
        print(f"Error during execution: {e}")
    
    print("\n\nDecoding done\n\n")


if __name__ == "__main__":
    main()
