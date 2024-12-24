from transformers import Blip2ForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from peft import LoraConfig, get_peft_model

def load_model():
    """
    Load the BLIP-2 model with LoRA configurations and 8-bit quantization.
    """
    # Configure quantization to enable efficient memory usage
    quant_config = BitsAndBytesConfig(
        load_in_8bit=False,                      # Use 8-bit precision for model weights
        llm_int8_enable_fp32_cpu_offload=True  # Offload some operations to CPU if needed
    )

    # Load the processor for BLIP-2
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    # Load the BLIP-2 model with sharded weights and device mapping
    model = Blip2ForConditionalGeneration.from_pretrained(
        "ybelkada/blip2-opt-2.7b-fp16-sharded",
        device_map="auto",                     # Automatically map parts of the model to GPU/CPU
        quantization_config=quant_config       # Apply quantization config
    )

    # Configure LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=16,                                  # Rank of the low-rank matrices
        lora_alpha=32,                         # Scaling factor for LoRA layers
        lora_dropout=0.05,                     # Dropout rate for LoRA layers
        bias="none",                           # No bias modification
        target_modules=["q_proj", "k_proj"]    # Apply LoRA to specific modules
    )

    # Wrap the model with LoRA
    model = get_peft_model(model, lora_config)

    # Print the number of trainable parameters
    model.print_trainable_parameters()

    return model, processor