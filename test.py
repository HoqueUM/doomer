from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./doomer_detector",
    tokenizer="./doomer_detector"
)

print(classifier("The Great Resegregation"))