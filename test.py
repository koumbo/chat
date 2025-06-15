import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load chatbot model and tokenizer (fine-tuned or distilgpt2)
model_name = "chatbot_model" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load emotion detection model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

st.title("🤖 Emotion-Aware Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("You:", "")

if user_input:
    # Detect emotion
    emotion_result = emotion_classifier(user_input)[0]
    emotion = emotion_result['label']
    score = emotion_result['score']
    
    # Add to chat history with detected emotion
    st.session_state.chat_history.append(f"You ({emotion}): {user_input}")
    
    # Generate response with context + emotion
    context = "\n".join(st.session_state.chat_history)
   # prompt = context + f"\nBot ({emotion}):"
    prompt = f"As an empathetic chatbot, respond with {emotion} to:\nUser: {user_input}\nBot:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the latest bot reply
    response_lines = reply.split("\n")
    last_bot_reply = next((line for line in reversed(response_lines) if line.startswith("Bot")), "Bot: ...")
    bot_reply = last_bot_reply.replace("Bot", "").strip()

    # Add bot response to chat
    st.session_state.chat_history.append(f"Bot ({emotion}): {bot_reply}")

# Display chat history
for message in st.session_state.chat_history:
    st.write(message)
