# from google import drive
# import os

# from reply_generator import EmailGenerator
# from intent_classify import IntentClassifier

# def configg() : # --- Colab & Path Setup ---
#     print("Mounting Google Drive...")
#     drive.mount('/content/drive')

# project_folder = 'MailAssistant'
# drive_model_path = os.path.join("/content/drive/MyDrive/", project_folder)
# # Define model paths using os.path.join
# GEMMA_BASE_MODEL_ID = os.path.join(drive_model_path, "gemma-2b-it")
# GEMMA_ADAPTER_PATH = os.path.join(drive_model_path, "fine_tuned_gemma_2b_adapters")
# BERT_MODEL = os.path.join(drive_model_path, "mail_category")

# print(f"Base Model Path: {GEMMA_BASE_MODEL_ID}")
# print(f"Adapter Path: {GEMMA_ADAPTER_PATH}")
# print(f"BERT Model Path: {BERT_MODEL}")

import os

# --- Local Path Configuration ---
# Use relative paths from your project directory (D:\DS\Project TODO\Corporate - Mail Assitant)

# Path to the fine-tuned BERT model for classification
BERT_MODEL = "./mail_category"

# Path to the base Gemma model
GEMMA_BASE_MODEL_ID = "google/gemma-2b-it" 

# Path to your fine-tuned Gemma adapters
GEMMA_ADAPTER_PATH = "./fine_tuned_gemma_2b_adapters"

print("--- Configuration Loaded (Local Paths) ---")
print(f"BERT Model Path: {os.path.abspath(BERT_MODEL)}")
print(f"Gemma Base Model: {GEMMA_BASE_MODEL_ID}")
print(f"Gemma Adapter Path: {os.path.abspath(GEMMA_ADAPTER_PATH)}")