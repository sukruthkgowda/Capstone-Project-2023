from transformers import T5ForConditionalGeneration, T5Tokenizer
import random

model_path = "./newmodel"  # Replace with the path to your fine-tuned model
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Read the input text from a file named "input.txt"
with open("./uploads/input.txt", "r") as input_file:
    input_text = input_file.read()

# Encode the input text using your T5 tokenizer
input_ids = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True, padding=True)

# Generate the abstractive summary
summary_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

# Decode the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Save the summary to a file named "output.txt"
with open("./outputs/output.txt", "w") as output_file:
    string_list = ["Okay class, let's understand ", "Here, ", "So class, ", "Let's look at "]
    output_file.write(random.choice(string_list) + summary)