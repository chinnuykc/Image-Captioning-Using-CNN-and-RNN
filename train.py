import torch
from torch.utils.data import DataLoader
from model import load_model
from data import download_and_process_data, ImageCaptioningDataset
from transformers import AutoProcessor
from tqdm import tqdm

# Define training constants
EPOCHS = 200
BATCH_SIZE = 3
LEARNING_RATE = 5e-4

def main():
    """
    Main function to handle the image captioning training pipeline.
    1. Downloads and processes the dataset.
    2. Loads the model and processor.
    3. Sets up the dataloader, optimizer, and training loop.
    4. Saves the trained model and processor.
    """
    # Download and process dataset and load processor
    dataset, processor = download_and_process_data()
    train_dataset = ImageCaptioningDataset(dataset, processor)

    def collate_fn(batch):
        """
        Custom collate function to process a batch of data for the DataLoader.
        Args:
            batch (list): A list of samples where each sample contains image data and text.
        Returns:
            dict: A dictionary containing processed pixel values, input IDs, and attention masks.
        """
        # Stack image pixel values into a tensor
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        texts = [example["text"] for example in batch]

        # Tokenize text inputs using the processor's tokenizer
        text_inputs = processor.tokenizer(
            texts, padding=True, return_tensors="pt"
        )

        # Add the <image> token at the start of each sequence
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        input_ids_with_image = []
        attention_masks_with_image = []

        for input_ids, attention_mask in zip(text_inputs.input_ids, text_inputs.attention_mask):
            input_ids_with_image.append(
                torch.cat([torch.tensor([image_token_id]), input_ids])
            )
            attention_masks_with_image.append(
                torch.cat([torch.tensor([1]), attention_mask])
            )

        # Pad sequences to the same length
        input_ids_with_image = torch.nn.utils.rnn.pad_sequence(
            input_ids_with_image, batch_first=True, padding_value=processor.tokenizer.pad_token_id
        )
        attention_masks_with_image = torch.nn.utils.rnn.pad_sequence(
            attention_masks_with_image, batch_first=True, padding_value=0
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids_with_image,
            "attention_mask": attention_masks_with_image,
        }

    # Create a DataLoader with custom collate function
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    # Load the pre-trained model and processor
    model, processor = load_model()  # Load model and map parameters to the appropriate device

    # Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Start training loop
    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0.0

        for idx, batch in enumerate(train_dataloader):
            # Move inputs to the appropriate device
            input_ids = batch["input_ids"].to(model.device)
            pixel_values = batch["pixel_values"].to(model.device, dtype=torch.float16)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=input_ids  # Use input IDs as labels for auto-regressive tasks
            )
            loss = outputs.loss
            epoch_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Print loss for the current batch
            print(f"Batch {idx + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        # Print average loss for the epoch
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_dataloader):.4f}")

    print("Training completed successfully.")

    # Save the trained model and processor
    model.save_pretrained("./saved_model")
    processor.save_pretrained("./saved_model")
    print("Model and processor saved successfully.")

if __name__ == "__main__":
    main()
