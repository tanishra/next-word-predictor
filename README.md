# 🧠 Gutenberg Next Word Predictor

A deep learning-based **Next Word Prediction** model trained on the **first 10,000 lines of the Gutenberg dataset**, using LSTM neural networks. The project includes an interactive **Streamlit app** where users can input a sentence and receive the next 5 predicted words.

---

## 🚀 Demo

[Click here to try the app →](#) *(https://next-word-predictor-ht4w7x7avkqinfrjigrkti.streamlit.app)*

---

## 📚 Dataset

This model is trained on the first **10,000 lines** of the **Gutenberg dataset**, which consists of public domain books. The data is preprocessed and tokenized using Keras' `Tokenizer`.

---

## 🧰 Tech Stack

- Python 🐍
- TensorFlow / Keras 🔧
- LSTM (Recurrent Neural Network)
- Streamlit (for UI)
- Pickle (to save tokenizer)

---

## 🏗️ Model Architecture

- **Embedding Layer:** `(input_dim=617, output_dim=100)`
- **LSTM Layer:** 1 layer with 150 units 
- **Dense Output:** Softmax over 617 words
- **Loss Function:** `categorical_crossentropy`

> Final model is saved as `model.keras`, and tokenizer as `tokenizer.pkl`.

---

## 📦 Setup Instructions

1. **Clone the repo:**

   ```bash
   git clone https://github.com/tanishra/next-word-predictor.git
   cd next-word-predictor
   `````
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    `````
3. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    `````

---

## ⚠️ Disclaimer
- This model is trained only on the first 10,000 lines of the Gutenberg dataset.
- Predictions may not generalize well to modern or out-of-domain text.

--- 

## 🧠 Future Work

This project is built for learning and experimentation purposes, using a basic LSTM model and a small subset of data. Future improvements may include:

- Train on the full Gutenberg corpus
- Add beam search for better predictions
- Replace current model with a more powerful architecture (e.g., Transformer, GPT-like models)

---

## 🤝 Contribution

Contributions and suggestions are always welcome! 
- Feel free to fork the repo, improve the code, fix bugs, or add new features. 
- Create a pull request whenever you're ready.

