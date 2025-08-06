import streamlit as st
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load model and tokenizer
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, output_attentions=True)
model.eval()

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs).item()
    sentiment = "Positive" if pred_class == 1 else "Negative"
    confidence = probs[0][pred_class].item()
    return sentiment, confidence, inputs, outputs.attentions

# Extract average attention from last layer
def get_attention(tokens, attentions):
    last_layer_attention = attentions[-1]  # [1, heads, tokens, tokens]
    mean_attention = last_layer_attention.mean(dim=1)[0]  # average heads → [tokens, tokens]
    return mean_attention.cpu(), tokens

# Plot heatmap
def plot_attention(tokens, attention):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention.numpy(), 
                xticklabels=tokens, 
                yticklabels=tokens, 
                cmap="viridis", 
                square=True, 
                linewidths=0.5,
                ax=ax)
    ax.set_title("Attention Heatmap (Last Layer, Averaged Heads)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    return fig

# Streamlit UI
st.set_page_config(page_title="Sentiment + Attention Visualizer", layout="wide")
st.title("Sentiment Analysis with Attention Heatmap")

user_input = st.text_area("Enter your review text:", height=150, placeholder="Type something like 'I loved the product, it works great!'")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        with st.spinner("Analyzing..."):
            sentiment, confidence, inputs, attentions = predict_sentiment(user_input)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            attention_matrix, token_list = get_attention(tokens, attentions)

        st.success(f"**Prediction:** {sentiment} ({confidence * 100:.2f}% confidence)")

        st.subheader("Attention Heatmap")
        st.markdown("Shows which tokens the model focused on when making the prediction.")
        fig = plot_attention(token_list, attention_matrix)
        st.pyplot(fig)
