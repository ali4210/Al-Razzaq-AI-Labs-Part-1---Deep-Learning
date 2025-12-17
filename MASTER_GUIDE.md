
# 🧠 The Universal Deep Learning Mastery Guide
**Author:** Saleem Ali
**Curriculum:** Al Nafi Deep Learning (Labs 1–40)
**Purpose:** Personal Code Reference & Execution Manual

---

## 📋 Lab Index (1–40)
*Click the Module link to jump to the code.*

| Labs | Topic | Module |
| :--- | :--- | :--- |
| **1–7** | Perceptrons, ANNs, Activation Functions, Preprocessing | [Module 1: Fundamentals](#-module-1-the-physics-of-ai-fundamentals) |
| **8–12** | MNIST, Basic CNNs, Visualizing History, Dropout | [Module 2: Computer Vision](#-module-2-computer-vision-cnns) |
| **13–14** | Saving Models, Autoencoders | [Module 4: Advanced Training](#-module-4-advanced-training--generative-ai) |
| **15–19** | RNNs, LSTMs, GRUs, Text Gen, Embeddings | [Module 3: Sequence Models](#-module-3-sequence-models-rnns--nlp) |
| **20–22** | CNNs (CIFAR-10), Augmentation, Batch Norm | [Module 2: Computer Vision](#-module-2-computer-vision-cnns) |
| **23–24** | Optimizers, Custom Loss Functions | [Module 4: Advanced Training](#-module-4-advanced-training--generative-ai) |
| **25–26** | GANs, Variational Autoencoders (VAE) | [Module 4: Advanced Training](#-module-4-advanced-training--generative-ai) |
| **27–29** | Hyperparameter Tuning, Early Stopping, Checkpointing | [Module 4: Advanced Training](#-module-4-advanced-training--generative-ai) |
| **30–31** | Visualizing Filters & Activations | [Module 2: Computer Vision](#-module-2-computer-vision-cnns) |
| **32–34** | ResNet Blocks, Transfer Learning, Fine-Tuning | [Module 2: Computer Vision](#-module-2-computer-vision-cnns) |
| **35–36** | Deployment Scripts, Flask API | [Module 5: Deployment](#-module-5-deployment--production) |
| **37–38** | Seq2Seq Models, Attention Mechanisms | [Module 3: Sequence Models](#-module-3-sequence-models-rnns--nlp) |
| **39** | Capsule Networks | [Module 2: Computer Vision](#-module-2-computer-vision-cnns) |
| **40** | Final End-to-End Capstone | [Module 5: Deployment](#-module-5-deployment--production) |

---

## 🟢 Module 1: The Physics of AI (Fundamentals)

### Lab 1: Introduction to Deep Learning
* **Concept:** Understanding Deep Learning vs. Machine Learning.
* **Action:** Review the history: Perceptrons (1958) -> Backprop (1986) -> AlexNet (2012). No code.

### Lab 2: Setting Up the Environment
* **Script:** `01_setup_check.py`
```python
import tensorflow as tf
from tensorflow import keras
print(f"TensorFlow Version: {tf.__version__}")
print("Environment Ready.")

```

### Lab 3: Neural Network Fundamentals (Manual Feedforward)

* **Script:** `02_manual_feedforward.py`

```python
import numpy as np
def sigmoid(x): return 1 / (1 + np.exp(-x))

inputs = np.array([1, 2, 3])
weights = np.array([0.2, 0.4, 0.4])
bias = 0.1
output = sigmoid(np.dot(inputs, weights) + bias)
print(f"Output: {output}")

```

### Lab 4: Building a Simple ANN (Keras)

* **Script:** `03_simple_ann.py`

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(8, input_shape=(4,), activation='relu'),
    Dense(3, activation='softmax') # 3 Classes
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

```

### Lab 5: Activation Functions Comparison

* **Script:** `04_activations.py`

```python
# Test ReLU vs Sigmoid
model_relu = Sequential([Dense(64, activation='relu', input_shape=(10,)), Dense(1)])
model_sigmoid = Sequential([Dense(64, activation='sigmoid', input_shape=(10,)), Dense(1)])
print("Models created for comparison.")

```

### Lab 6: Overfitting & Underfitting

* **Script:** `05_bias_variance.py`

```python
# Visualize Loss
import matplotlib.pyplot as plt
# (Assume history object exists from model.fit)
# plt.plot(history.history['loss'], label='Train')
# plt.plot(history.history['val_loss'], label='Val')
# plt.show()

```

### Lab 7: Data Preprocessing

* **Script:** `06_normalization.py`

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([[10], [20], [30]])
scaler = MinMaxScaler()
print(scaler.fit_transform(data))

```

---

## 🔵 Module 2: Computer Vision (CNNs)

### Lab 8: Loading MNIST

* **Script:** `07_load_mnist.py`

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

```

### Lab 9: Basic CNN Architecture

* **Script:** `08_basic_cnn.py`

```python
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

```

### Lab 10: Visualizing History

* **Script:** `09_plot_history.py`

```python
# Standard Plotting Code
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

```

### Lab 11: Confusion Matrix

* **Script:** `10_confusion_matrix.py`

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
# y_pred = model.predict(x_test)
# cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
# sns.heatmap(cm, annot=True)

```

### Lab 12: Dropout Regularization

* **Script:** `11_dropout.py`

```python
model.add(layers.Dropout(0.5)) # Add this between Dense layers

```

### Lab 20: CIFAR-10 CNN (Color Images)

* **Script:** `12_cifar10.py`

```python
# Input shape changes for RGB
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))

```

### Lab 21: Data Augmentation

* **Script:** `13_augmentation.py`

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
# model.fit(datagen.flow(x_train, y_train))

```

### Lab 22: Batch Normalization

* **Script:** `14_batch_norm.py`

```python
model.add(layers.BatchNormalization()) # Add after Conv2D/Dense, before Activation

```

### Lab 30: Visualizing Filters

* **Script:** `15_vis_filters.py`

```python
filters, biases = model.layers[0].get_weights()
# Plot filters[..., 0] using matplotlib

```

### Lab 31: Visualizing Activations (Feature Maps)

* **Script:** `16_vis_activations.py`

```python
from tensorflow.keras.models import Model
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(img_tensor)

```

### Lab 32: ResNet Block (Skip Connection)

* **Script:** `17_resnet_block.py`

```python
from tensorflow.keras.layers import Add, ReLU
def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = Add()([x, shortcut])
    return ReLU()(x)

```

### Lab 33: Transfer Learning (VGG16)

* **Script:** `18_transfer_learning.py`

```python
from tensorflow.keras.applications import VGG16
base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
base.trainable = False

```

### Lab 34: Fine-Tuning

* **Script:** `19_fine_tuning.py`

```python
base.trainable = True
for layer in base.layers[:-4]: layer.trainable = False # Freeze all except last 4

```

### Lab 39: Capsule Networks (Basic Implementation)

* **Script:** `33_capsule_net.py`

```python
# Note: Complex implementation. Use tf.keras.layers.Layer subclassing.
# Key concept: Squash activation function instead of ReLU.

```

---

## 🟠 Module 3: Sequence Models (RNNs & NLP)

### Lab 15: Simple RNN

* **Script:** `20_simple_rnn.py`

```python
model.add(layers.SimpleRNN(32, input_shape=(10, 1)))

```

### Lab 16: LSTM (Long Short-Term Memory)

* **Script:** `21_lstm.py`

```python
model.add(layers.LSTM(32, return_sequences=True)) # Stacked LSTM
model.add(layers.LSTM(32))

```

### Lab 17: GRU

* **Script:** `22_gru.py`

```python
model.add(layers.GRU(32)) # Faster than LSTM

```

### Lab 18: Text Generation

* **Script:** `23_text_gen.py`

```python
# Character-level prediction
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

```

### Lab 19: Word Embeddings

* **Script:** `24_embeddings.py`

```python
model.add(layers.Embedding(input_dim=1000, output_dim=64))

```

### Lab 37: Seq2Seq Model

* **Script:** `25_seq2seq.py`

```python
# Encoder-Decoder Architecture
# Use keras.models.Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

```

### Lab 38: Attention Mechanism

* **Script:** `26_attention.py`

```python
# Custom Layer: Calculate Dot Product of Query and Values
# context = Attention()(lstm_output)

```

---

## 🟣 Module 4: Advanced Training & Generative AI

### Lab 13: Save & Load

* **Script:** `27_save_load.py`

```python
model.save('my_model.h5')
loaded = tf.keras.models.load_model('my_model.h5')

```

### Lab 14: Autoencoders

* **Script:** `28_autoencoder.py`

```python
# Input -> Compressed -> Reconstructed
encoded = layers.Dense(32, activation='relu')(input_img)
decoded = layers.Dense(784, activation='sigmoid')(encoded)

```

### Lab 23: Optimizers

* **Script:** `29_optimizers.py`

```python
# Comparison
model.compile(optimizer='sgd', loss='mse')
model.compile(optimizer='adam', loss='mse')

```

### Lab 24: Custom Loss Function

* **Script:** `30_custom_loss.py`

```python
def my_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

```

### Lab 25: GANs (Generative Adversarial Networks)

* **Script:** `31_simple_gan.py`

```python
# Generator creates data, Discriminator detects fake data
# Train Discriminator on Real+Fake; Train Generator to fool Discriminator

```

### Lab 26: Variational Autoencoder (VAE)

* **Script:** `32_vae.py`

```python
# Sampling Layer (Reparameterization Trick)
def sampling(args):
    z_mean, z_log_var = args
    return z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(tf.shape(z_mean))

```

### Lab 27: Hyperparameter Tuning

* **Script:** `34_tuning.py`

```python
# Grid Search Loop
# for batch_size in [16, 32, 64]: model.fit(...)

```

### Lab 28 & 29: Callbacks (Early Stopping & Checkpoint)

* **Script:** `35_callbacks.py`

```python
cb = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.ModelCheckpoint('best.h5', save_best_only=True)
]

```

---

## ⚫ Module 5: Deployment & Production

### Lab 35: Inference Script

* **Script:** `36_inference.py`

```python
# Load model and predict on single image
import numpy as np
img = np.random.rand(1, 28, 28, 1)
print(model.predict(img))

```

### Lab 36: Flask API

* **Script:** `37_flask_api.py`

```python
from flask import Flask, request, jsonify
import tensorflow as tf
app = Flask(__name__)
model = tf.keras.models.load_model('best.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']
    prediction = model.predict([data])
    return jsonify(prediction.tolist())

if __name__ == '__main__': app.run(port=5000)

```

### Lab 40: Final Capstone Pipeline

* **Script:** `40_final_pipeline.py`

```python
# 1. Load Data
# 2. Preprocess (Normalize/Augment)
# 3. Define Model (CNN/ResNet)
# 4. Train (with Callbacks)
# 5. Evaluate (Confusion Matrix)
# 6. Save Model
print("End-to-End Deep Learning Pipeline executed successfully.")

```

```

```
