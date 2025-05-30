import streamlit as st
from chatbot.core import EcommerceAssistant, chatbot_response, clear_conversation
import time

# Page configuration
st.set_page_config(
    page_title="E-commerce Assistant",
    page_icon="🛍️",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "assistant" not in st.session_state:
    st.session_state.assistant = None

# Initialize the assistant without cache to control the spinner
def create_assistant():
    if st.session_state.assistant is None:
        with st.spinner("Creating assistant and loading RAG, please wait..."):
            st.session_state.assistant = EcommerceAssistant()
    return st.session_state.assistant

# Title and description
st.title("🛍️ E-commerce Assistant")
st.markdown("""
    Welcome to the e-commerce assistant. You can ask questions about:
    - Products and recommendations
    - Order status
    - Category information
    - And more...
""")

# Show the message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What can I help you with today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant = create_assistant()
            response = chatbot_response(prompt, assistant)
            st.markdown(response)
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with additional information
with st.sidebar:
    st.header("📝 Example Questions")
    st.markdown("""
    You can ask things like:
    - What are the best-rated guitars?
    - What is the status of my order? (ID: 37077)
    - I don't know the date of my recent order
    - Show me the most popular microphone
    """)
    
    # Clear conversation button
    if st.button("Clear conversation"):
        st.session_state.messages = []
        if st.session_state.assistant:
            clear_conversation(st.session_state.assistant)
        st.rerun() 