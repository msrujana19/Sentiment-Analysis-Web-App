# Sentiment Analysis Web App

A Streamlit-powered web application that classifies **e-commerce product reviews** as **Positive** or **Negative**, with attention heatmap visualization to explain model behavior.

## Features

- **Sentiment Classification** using Hugging Face Transformer models (`distilbert-base-uncased-finetuned-sst-2-english`)
- **Attention Heatmap Visualization** to understand what words influenced the model
- Clean and interactive **Streamlit UI**
- Ready to be extended with Aspect-Based Sentiment Analysis (ABSA) and recommendations

---



---

## Tech Stack

- Python 3.10+
- [Streamlit](https://streamlit.io/)
- [Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index)
- PyTorch
- Seaborn / Matplotlib (for visualization)

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-app.git
cd sentiment-analysis-app

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
