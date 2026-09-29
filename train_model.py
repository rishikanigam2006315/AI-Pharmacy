import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# ==========================================
# 1. SETTINGS
# ==========================================

DATASET_PATH = Path("dataset")

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 15

TRAIN_PATH = DATASET_PATH / "train"
VALIDATION_PATH = DATASET_PATH / "validation"

MODEL_PATH = "tablet_model.keras"


# ==========================================
# 2. CHECK DATASET
# ==========================================

print("\n===================================")
print("   AI PHARMACY - TABLET AI MODEL")
print("===================================\n")

print("Checking dataset...")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        "Training folder not found: dataset/train"
    )

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(
        "Validation folder not found: dataset/validation"
    )

print("Dataset found successfully!\n")


# ==========================================
# 3. LOAD TRAINING DATA
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


# ==========================================
# 4. LOAD VALIDATION DATA
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 5. CLASS NAMES
# ==========================================

class_names = train_dataset.class_names

print("\nClasses detected:")
print(class_names)

# Expected:
# ['defective', 'normal']


# ==========================================
# 6. PERFORMANCE OPTIMIZATION
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# 7. DATA AUGMENTATION
# ==========================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")


# ==========================================
# 8. LOAD PRE-TRAINED MOBILENETV2
# ==========================================

print("\nLoading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

print("MobileNetV2 loaded successfully!")


# ==========================================
# 9. BUILD AI MODEL
# ==========================================

inputs = keras.Input(
    shape=(224, 224, 3)
)

# Data augmentation
x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Feature extraction
x = base_model(
    x,
    training=False
)

# Convert feature maps to vector
x = layers.GlobalAveragePooling2D()(x)

# Prevent overfitting
x = layers.Dropout(0.30)(x)

# Binary classification
outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(
    inputs,
    outputs
)


# ==========================================
# 10. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 11. SHOW MODEL
# ==========================================

print("\nModel Summary:\n")

model.summary()


# ==========================================
# 12. TRAIN MODEL
# ==========================================

print("\n===================================")
print("       STARTING AI TRAINING")
print("===================================\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ==========================================
# 13. EVALUATE MODEL
# ==========================================

print("\n===================================")
print("       MODEL EVALUATION")
print("===================================\n")

loss, accuracy = model.evaluate(
    validation_dataset
)

print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 14. SAVE MODEL
# ==========================================

model.save(MODEL_PATH)

print("\n===================================")
print("       MODEL SAVED SUCCESSFULLY")
print("===================================\n")

print(f"Model file: {MODEL_PATH}")


# ==========================================
# 15. SAVE CLASS NAMES
# ==========================================

with open("class_names.txt", "w") as file:
    for class_name in class_names:
        file.write(class_name + "\n")

print("Class names saved: class_names.txt")

print("\nTraining completed successfully! 🎉")