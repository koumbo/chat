import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model
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

# Title
st.title(" Chat with bot")

# Chat history session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores (user, bot) pairs

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", "")

# Handle input and response
if user_input:
    # Build conversation prompt from history
    history_text = ""
    for u, b in st.session_state.chat_history:
        history_text += f"User: {u}{tokenizer.eos_token}Bot: {b}{tokenizer.eos_token}"
    history_text += f"User: {user_input}{tokenizer.eos_token}Bot:"

    # Tokenize with context
    inputs = tokenizer(history_text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the new response
    bot_response = full_output.split("Bot:")[-1].strip()

    # Update memory
    st.session_state.chat_history.append((user_input, bot_response))

# Show full conversation
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**🤖 Bot:** {bot_msg}")
