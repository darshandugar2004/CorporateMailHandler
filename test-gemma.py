import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
# from trl import SFTTrainer
# from huggingface_hub import login
import os
# from google.colab import userdata


# HF_TOKEN = userdata.get('HF_TOKEN')
# login(token=HF_TOKEN)

# --- 3. Configuration ---
# CHANGED: Model ID is now Gemma-2B
model_id = "D:\DS\Project TODO\Corporate - Mail Assitant\google-gemma-2b-it"
output_dir = "./fine_tuned_gemma_2b_adapters"


# --- 4. Define Quantization Configuration ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- 5. Load Model and Tokenizer ---
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    # use_auth_token=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set padding token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Prepare a Test Prompt ---
prompt = "As a corporate assistant, write a formal email based on the following intent and details. Intent: Merger Announcement. Details: Verification API: FAILED - Inconsistent company registration number. Ticket #T9999 raised for manual review."
# Use the Gemma-specific prompt format
formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

# --- 5. Generate the Response ---
# Ensure your device has CUDA available for this to work
if torch.cuda.is_available():
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = response_text.split("<start_of_turn>model\n")[-1]

    print("--- RESPONSE FROM BASE GEMMA-2B MODEL (BEFORE TRAINING) ---")
    print(completion)
else:
    print("CUDA is not available. Please run on a GPU-enabled machine.")