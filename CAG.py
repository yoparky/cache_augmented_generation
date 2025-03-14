import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache
from accelerate.test_utils.testing import get_backend
from huggingface_hub import login

login(token = "???")

# Init StaticCache with big enough max-length (1024 tokens for the below example) 
# prompt_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=1024, device="cuda", dtype=torch.bfloat16)
class CAG:
    def __init__(self, model_id: str, initial_prompt: str, max_new_tokens: int = 20):
        self.device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.initial_prompt = initial_prompt
        self.max_new_tokens = max_new_tokens
        self.initial_prompt_token_count = 0

        # Initialize the prompt cache
        self.prompt_cache = DynamicCache()
        self.populate_kv_cache(initial_prompt)
    
    def __eq__(self, other):
        return self.model == other.model and self.tokenizer == other.tokenizer and self.initial_prompt == other.initial_prompt and self.prompt_cache == other.prompt_cache
    
    def __hash__(self):
        return hash((self.model, self.tokenizer, self.initial_prompt, self.prompt_cache))
    
    def close(self):
        """Explicitly clean up resources."""
        if hasattr(self, 'prompt_cache'):
            self.prompt_cache = None
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        self.close()
    
    def clear_prompt_cache(self):
        self.prompt_cache = DynamicCache()
        torch.cuda.empty_cache()

    def populate_kv_cache(self, initial_prompt: str):
        try:
            inputs_initial_prompt = self.tokenizer(initial_prompt, return_tensors="pt").to(self.device)
            self.initial_prompt_token_count = inputs_initial_prompt.input_ids.shape[1]
            # This is the common prompt cached
            with torch.no_grad():
                self.prompt_cache = self.model(**inputs_initial_prompt, past_key_values = self.prompt_cache).past_key_values
        except Exception as e:
            print(f"Error populating kv cache: {e}")
            self.clear_prompt_cache()

    def cag_decode_batch(self, prompts: list[str]):
        if not prompts: return []
        
        responses = []
        for prompt in prompts:
            try:
                new_inputs = self.tokenizer(self.initial_prompt + prompt, return_tensors="pt").to(self.device)
                input_token_length = new_inputs.input_ids.shape[1]

                past_key_values = copy.deepcopy(self.prompt_cache)
                outputs = self.model.generate(**new_inputs, past_key_values=past_key_values,max_new_tokens=self.max_new_tokens,return_dict_in_generate=True, output_scores=False)
                new_tokens = outputs.sequences[0, input_token_length:]
                response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
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
                new_inputs = self.tokenizer(self.initial_prompt + prompt, return_tensors="pt").to(self.device)
                input_token_length = new_inputs.input_ids.shape[1]

                past_key_values = copy.deepcopy(self.prompt_cache)
                outputs = self.model.generate(
                    **new_inputs, 
                    past_key_values=past_key_values,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True, 
                    output_scores=False,
                    # generation control
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.1,
                )
                
                new_tokens = outputs.sequences[0, input_token_length:]
                
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                responses.append(response)
            except Exception as e:
                print(f"Error decoding prompt: {e}")
                responses.append(None)
        return responses

def main():
    """Example usage of the CAG class."""
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    initial_prompt = """
        PROLOGUE, A Game of Thrones
    We should start back,” Gared urged as the woods began to grow dark around them. “The
    wildlings are dead.”
    “Do the dead frighten you?” Ser Waymar Royce asked with just the hint of a smile.
    Gared did not rise to the bait. He was an old man, past fifty, and he had seen the
    lordlings come and go. “Dead is dead,” he said. “We have no business with the dead.”
    “Are they dead?” Royce asked softly. “What proof have we?”
    “Will saw them,” Gared said. “If he says they are dead, that’s proof enough for me.”
    Will had known they would drag him into the quarrel sooner or later. He wished it had
    been later rather than sooner. “My mother told me that dead men sing no songs,” he put
    in.
    “My wet nurse said the same thing, Will,” Royce replied. “Never believe anything you
    hear at a woman’s tit. There are things to be learned even from the dead.” His voice
    echoed, too loud in the twilit forest.
    “We have a long ride before us,” Gared pointed out. “Eight days, maybe nine. And night
    is falling.”
    Ser Waymar Royce glanced at the sky with disinterest. “It does that every day about this
    time. Are you unmanned by the dark, Gared?”
    Will could see the tightness around Gared’s mouth, the barely suppressed anger in his
    eyes under the thick black hood of his cloak. Gared had spent forty years in the Night’s
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
    leather. Ser Waymar had been a Sworn Brother of the Night’s Watch for less than half a
    year, but no one could say he had not prepared for his vocation. At least insofar as his
    wardrobe was concerned.
    His cloak was his crowning glory; sable, thick and black and soft as sin. “Bet he killed
    them all himself, he did,” Gared told the barracks over wine, “twisted their little heads
    off, our mighty warrior.” They had all shared the laugh.
    It is hard to take orders from a man you laughed at in your cups, Will reflected as he sat
    shivering atop his garron. Gared must have felt the same.
    “Mormont said as we should track them, and we did,” Gared said. “They’re dead. They
    shan’t trouble us no more. There’s hard riding before us. I don’t like this weather. If it
    snows, we could be a fortnight getting back, and snow’s the best we can hope for. Ever
    seen an ice storm, my lord?”
    The lordling seemed not to hear him. He studied the deepening twilight in that half-
    bored, half-distracted way he had. Will had ridden with the knight long enough to
    understand that it was best not to interrupt him when he looked like that. “Tell me again
    what you saw, Will. All the details. Leave nothing out.”
    Will had been a hunter before he joined the Night’s Watch. Well, a poacher in truth.
    Mallister freeriders had caught him red-handed in the Mallisters’ own woods, skinning
    one of the Mallisters’ own bucks, and it had been a choice of putting on the black or
    losing a hand. No one could move through the woods as silent as Will, and it had not
    taken the black brothers long to discover his talent.
    “The camp is two miles farther on, over that ridge, hard beside a stream,” Will said. “I
    got close as I dared. There’s eight of them, men and women both. No children I could
    see. They put up a lean-to against the rock. The snow’s pretty well covered it now, but I
    could still make it out. No fire burning, but the firepit was still plain as day. No one
    moving. I watched a long time. No living man ever lay so still.”
    “Did you see any blood?”
    “Well, no,” Will admitted.
    “Did you see any weapons?”
    “Some swords, a few bows. One man had an axe. Heavy-looking, double-bladed, a cruel
    piece of iron. It was on the ground beside him, right by his hand.”
    “Did you make note of the position of the bodies?”
    Will shrugged. “A couple are sitting up against the rock. Most of them on the ground.
    Fallen, like.”
    “Or sleeping,” Royce suggested.
    “Fallen,” Will insisted. “There’s one woman up an ironwood, half-hid in the branches. A
    far-eyes.” He smiled thinly. “I took care she never saw me. When I got closer, I saw that
    she wasn’t moving neither.” Despite himself, he shivered.
    “You have a chill?” Royce asked.
    “Some,” Will muttered. “The wind, m’lord.”
    The young knight turned back to his grizzled man-at-arms. Frostfallen leaves whispered
    past them, and Royce’s destrier moved restlessly. “What do you think might have killed
    these men, Gared?” Ser Waymar asked casually. He adjusted the drape of his long sable
    cloak.
    “It was the cold,” Gared said with iron certainty. “I saw men freeze last winter, and the
    one before, when I was half a boy. Everyone talks about snows forty foot deep, and how
    the ice wind comes howling out of the north, but the real enemy is the cold. It steals up
    on you quieter than Will, and at first you shiver and your teeth chatter and you stamp
    your feet and dream of mulled wine and nice hot fires. It burns, it does. Nothing burns
    like the cold. But only for a while. Then it gets inside you and starts to fill you up, and
    after a while you don’t have the strength to fight it. It’s easier just to sit down or go to
    sleep. They say you don’t feel any pain toward the end. First you go weak and drowsy,
    and everything starts to fade, and then it’s like sinking into a sea of warm milk. Peaceful,
    like.”
    “Such eloquence, Gared,” Ser Waymar observed. “I never suspected you had it in you.”
    “I’ve had the cold in me too, lordling.” Gared pulled back his hood, giving Ser Waymar a
    good long look at the stumps where his ears had been. “Two ears, three toes, and the
    little finger off my left hand. I got off light. We found my brother frozen at his watch,
    with a smile on his face.”
    Ser Waymar shrugged. “You ought dress more warmly, Gared.”
    Gared glared at the lordling, the scars around his ear holes flushed red with anger where
    Maester Aemon had cut the ears away. “We’ll see how warm you can dress when the
    winter comes.” He pulled up his hood and hunched over his garron, silent and sullen.
    “If Gared said it was the cold . . . ” Will began.
    """
    
    try:
        # Use context manager for automatic cleanup
        with CAG(model_id, initial_prompt, max_new_tokens=100) as cag:
            prompts = [
                "What happens in the prologue of A Game of Thrones?",
                "Who is Ser Waymar?"
            ]
            
            responses = cag.cag_decode(prompts)
            
            for response in responses:
                print(response)
    except Exception as e:
        print(f"Error during execution: {e}")
    
    print("\n\nDecoding done\n\n")


if __name__ == "__main__":
    main()
