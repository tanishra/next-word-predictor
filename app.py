import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load model
model = tf.keras.models.load_model('model.keras')

# Max length 
MAX_SEQUENCE_LEN = 17  

def predict_next_words(seed_text, num_words=5):
    output_text = seed_text

    for _ in range(num_words):
        # Tokenize input
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = token_list[-MAX_SEQUENCE_LEN:]  
        padded = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=MAX_SEQUENCE_LEN, padding='pre')

        # Predict next word
        predicted_probs = model.predict(padded, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]

        # Convert index to word
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_text += ' ' + word
                break
    return output_text

# Streamlit UI
st.set_page_config(page_title="Next Word Predictor", page_icon="🧠")

st.title("🧠 Next Word Predictor App")

# 🔽 Dataset disclaimer
st.markdown(
    """
    ⚠️ **Note:** This model is trained only on the first **10,000 lines** of the **Gutenberg dataset**.  
    Predictions may not generalize well beyond this data.
    """,
    unsafe_allow_html=True
)

st.write("Enter a seed sentence and get next 5 predicted words:")

# User input
seed_text = st.text_input("Enter your text here:", "")

# Predict button
if st.button("Predict"):
    if seed_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        predicted_text = predict_next_words(seed_text)
        st.success(f"Predicted Text: {predicted_text}")