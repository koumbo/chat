import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("chatbot_model")
model = GPT2LMHeadModel.from_pretrained("chatbot_model")
model.eval()

# Set page title
st.title("Memory-Enhanced Chatbot ")

# Initialize session state for conversation memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add a Clear Chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", "")

# Process input and update chat
if user_input:
    # Add user input to history
    st.session_state.chat_history.append(f"User: {user_input}")

    # Build the conversation context
    full_prompt = "\n".join(st.session_state.chat_history) + "\nBot:"

    # Tokenize and generate a response
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # Decode and save bot response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_reply = response.split("Bot:")[-1].strip()
    st.session_state.chat_history.append(f"Bot: {bot_reply}")

# Display the conversation
st.markdown("### Chat History")
for line in st.session_state.chat_history:
    st.write(line)
