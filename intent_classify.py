import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



class IntentClassifier:
    """
    A reusable classifier that loads a fine-tuned model once for efficient, repeated predictions.
    """
    def __init__(self, model_path: str):
        """
        Initializes the classifier by loading the tokenizer and model.

        Args:
            model_path (str): The path to the saved fine-tuned model and tokenizer.
        """
        # Set the device to GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load the tokenizer and model from the specified path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Move the model to the selected device
        self.model.to(self.device)
        
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Get the label mapping from the model's configuration
        self.id_to_label = self.model.config.id2label

    def predict(self, text: str) -> dict:
        """
        Classifies the intent of a single text string.

        Args:
            text (str): The input text to classify.

        Returns:
            dict: A dictionary containing the predicted 'label' and its 'confidence' score.
        """
        # Tokenize the input text and return PyTorch tensors
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        # Perform inference without calculating gradients to save memory and speed up
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the model's predictions (logits)
        logits = outputs.logits
        
        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1)
        
        # Get the top prediction's confidence and class ID
        confidence, predicted_id_tensor = torch.max(probabilities, dim=1)
        
        # Convert tensor results to standard Python types
        predicted_id = predicted_id_tensor.item()
        confidence_score = confidence.item()
        
        # Map the predicted ID to its corresponding label string
        predicted_label = self.id_to_label[predicted_id]

        return {"label": predicted_label, "confidence": confidence_score}

# --- HOW TO USE THE CLASS ---
if __name__ == "__main__":
    # 1. Define the path to your saved model
    save_path = "./mail_category"
    
    # 2. Create an instance of the classifier (this loads the model once)
    print("--- Loading model ---")
    classifier = IntentClassifier(model_path=save_path)
    print("--- Model loaded successfully ---\n")

    # 3. Use the classifier to predict intents for various texts
    text_to_classify_1 = "Can you send over the latest financial performance report for Q3?"
    prediction_1 = classifier.predict(text_to_classify_1)
    
    print(f"Text: '{text_to_classify_1}'")
    print(f"Predicted Intent: {prediction_1['label']} (Confidence: {prediction_1['confidence']:.4f})\n")

    text_to_classify_2 = "We are excited to announce our upcoming merger with Tech Solutions Inc."
    prediction_2 = classifier.predict(text_to_classify_2)

    print(f"Text: '{text_to_classify_2}'")
    print(f"Predicted Intent: {prediction_2['label']} (Confidence: {prediction_2['confidence']:.4f})\n")
