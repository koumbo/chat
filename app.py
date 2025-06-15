
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("fine_tuned_distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

st.title("Chat with our bot")
user_input = st.text_input("You:", "")

# Initialize session state for conversation memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Add a Clear Chat button
if st.button(" Clear Chat"):
    st.session_state.chat_history = []

if user_input:
    inputs = tokenizer(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(f"🤖: {response}")
