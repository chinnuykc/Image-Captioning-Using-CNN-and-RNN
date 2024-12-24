from transformers import Blip2ForConditionalGeneration, Blip2Processor
from PIL import Image
import torch

def load_trained_model():
    # Correct the path to your saved model
    model = Blip2ForConditionalGeneration.from_pretrained("saved_model")  # Path to your model
    processor = Blip2Processor.from_pretrained("saved_model")  # Path to your processor
    return model, processor

def generate_caption(image_path):
    model, processor = load_trained_model()

    # Load and process the image
    image = Image.open(image_path).convert("RGB")

    # Process the image
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        out = model.generate(**inputs)
    
    # Decode the caption
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    # Provide the correct path to your image
    image_path = "boat.png"  # Replace with your image path
    print("Generating caption for the image...")
    
    # Generate and print caption
    caption = generate_caption(image_path)
    print(f"Generated Caption: {caption}")
