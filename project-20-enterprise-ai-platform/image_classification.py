from tensorflow.keras.applications import ResNet50

model = ResNet50(
    weights="imagenet"
)

print(
    "Vision model loaded."
)
