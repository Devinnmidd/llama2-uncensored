import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "your-model-name"  # replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Sample input
input_text = "Hello, how are you?"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
outputs = model.generate(inputs['input_ids'])

# Decode and print output
print(tokenizer.decode(outputs[0]))

