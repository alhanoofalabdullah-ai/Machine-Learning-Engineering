from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis"
)

result = classifier(
    "Enterprise AI platform is amazing."
)

print(result)
