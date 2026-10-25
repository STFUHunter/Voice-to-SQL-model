import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

#since this project is local, and I don't have a crazy GPU(RTX 3050, GTX 1060 both with 6 gb VRAM)

#the true 
model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

#I will config the model with 4-bit Quantization (QLoRA)
bnbConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

#load model with the 4-bit config
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnbConfig,
    device_map = 'auto'
)

#preaparing for training
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r = 8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()