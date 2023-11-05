from datasets import load_dataset
import json

dataset = load_dataset("ncbi_disease")



# Access the dataset splits (e.g., "train", "test", "validation")
train_dataset = dataset["train"]
test_dataset = dataset["test"]
validation_dataset = dataset["validation"]

print(len(train_dataset))

print("===============================")

# Example: Print the first few examples from the training dataset
for example in train_dataset[:5]:
    print(example)
    
print("===============================")

# for sentence in train_dataset:
#     print(sentence)


# # Example: Access specific fields from an example
# for example in train_dataset:
#     text = example["tokens"]
#     labels = example["ner_tags"]
#     print(f"Text: {text}")
#     print(f"Labels: {labels}")


# Specify the split you want to download (e.g., "train")
split_name = "train"

# Access the specified split
train_dataset = dataset[split_name]

# Convert the dataset to a list of dictionaries
examples = [example for example in split_dataset]

# Define the filename where you want to save the dataset
output_filename = "train.json"

# Save the dataset to the file
with open(output_filename, "w") as output_file:
    json.dump(examples, output_file)

# print(examples)


# Specify the split you want to download (e.g., "train")
split_name = "test"

# Access the specified split
test_dataset = dataset[split_name]

# Convert the dataset to a list of dictionaries
examples = [example for example in test_dataset]

# Define the filename where you want to save the dataset
output_filename = "test.json"

# Save the dataset to the file
with open(output_filename, "w") as output_file:
    json.dump(examples, output_file)



# Specify the split you want to download (e.g., "train")
split_name = "validation"

# Access the specified split
validation_dataset = dataset[split_name]

# Convert the dataset to a list of dictionaries
examples = [example for example in validation_dataset]

# Define the filename where you want to save the dataset
output_filename = "validation.json"

# Save the dataset to the file
with open(output_filename, "w") as output_file:
    json.dump(examples, output_file)