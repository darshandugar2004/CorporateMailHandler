import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig
from peft import PeftModel

class EmailGenerator:

    def __init__(self, base_model_id: str, adapter_path: str):
        """
        Initializes the generator by loading the base model and merging LoRA adapters.

        Args:
            base_model_id (str): The identifier of the base model (e.g., 'google/gemma-2b-it').
            adapter_path (str): The local path to the fine-tuned LoRA adapters.
        """
        print("--- Initializing Email Generator ---")
        
        # 1. Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")


        # 4. Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        # 3. Load the base model
        print(f"Loading base model: {base_model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto", # for GPU only
            trust_remote_code=True,
        )

        # 5. Merge the LoRA adapters into the base model
        print(f"Loading and merging adapters from: {adapter_path}...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # 6. Set model to evaluation mode for faster inference
        self.model.eval()
        print("--- Initialization complete. Ready to generate. ---\n")

    def generate(self, intent: str, details: str) -> str:
        """
        Generates a formal email based on a given intent and details.

        Args:
            intent (str): The primary purpose of the email (e.g., 'Financial Report').
            details (str): Specific details to include in the email content.

        Returns:
            str: The generated email content.
        """
        # 1. Create a structured prompt for the model
        prompt = (
            "As a corporate assistant, write a formal email based on the following "
            f"intent and details.\n\nIntent: {intent}\n\nDetails: {details}"
        )
        
        # 2. Format the prompt according to Gemma's chat template
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        # 3. Tokenize the input and move it to the configured device
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # 4. Generate the response
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,    # Low temperature for professional, predictable output
                do_sample=True      # Enable sampling for more natural text
            )

        # 5. Decode the output, skipping special tokens
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 6. Clean the output to return only the model's generated text
        completion = response_text.split("<start_of_turn>model\n")[-1].strip()

        return completion

# ==============================================================================
# 🚀 HOW TO USE THE CLASS
# ==============================================================================
if __name__ == "__main__":
    # Define model paths
    BASE_MODEL_ID = "google/gemma-2b-it"
    ADAPTER_PATH = "./fine_tuned_gemma_2b_adapters"
    
    # 1. Create an instance of the generator.
    #    (This is the slow part that loads the model ONCE.)
    email_bot = EmailGenerator(base_model_id=BASE_MODEL_ID, adapter_path=ADAPTER_PATH)

    # 2. Define the email requirements.
    email_intent = "Financial Performance Summary"
    email_details = "Summarize the key findings from the Q3 financial report. Mention the 15% revenue growth, the successful launch of Project Phoenix, and the new projection of a 12% profit margin for Q4. The email should be addressed to all department heads."

    # 3. Generate the reply.
    #    (This part is fast and can be called repeatedly.)
    print("--- Generating Email ---")
    generated_email = email_bot.generate(intent=email_intent, details=email_details)
    
    # 4. Print the result.
    print("\n--- Generated Email ---\n")
    print(generated_email)

    # Example of a second, different request (fast because the model is already loaded)
    print("\n\n--- Generating Second Email ---")
    generated_email_2 = email_bot.generate(
        intent="Merger Announcement",
        details="Announce the successful merger with Innovate Corp. Emphasize the strategic benefits and the expected synergies. A town hall meeting is scheduled for next Friday at 10 AM."
    )
    print("\n--- Generated Email ---\n")
    print(generated_email_2)