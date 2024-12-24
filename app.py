import gradio as gr
import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from PIL import Image
import traceback
import numpy as np

def load_trained_model():
    try:
        model = Blip2ForConditionalGeneration.from_pretrained("saved_model")
        processor = Blip2Processor.from_pretrained("saved_model")
        
        # Move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        print(traceback.format_exc())
        raise

def generate_caption(image):
    try:
        model, processor = load_trained_model()
        
        # Convert image to PIL and ensure RGB
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        # Prepare inputs with explicit device handling
        inputs = processor(images=image, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with max length and some additional parameters
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_length=50,  # Limit max generation length
                num_beams=4,    # Use beam search
                early_stopping=True
            )
        
        # Decode the generated caption
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)}"

def predict(image):
    try:
        caption = generate_caption(image)
        return caption
    except Exception as e:
        print(f"Predict error: {e}")
        print(traceback.format_exc())
        return f"Prediction error: {str(e)}"

# Modify interface to show error messages
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload an Image:"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning with Fine-Tuned BLIP Model",
    description="Upload an image and get the caption"
)

if __name__ == "__main__":
    interface.launch()