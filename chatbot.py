import streamlit as st
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from streamlit_chat import message
import uuid  # Import UUID to generate unique keys

# Set up Streamlit app
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot powered by Llama 2")

# Initialize Ollama model
llm = Ollama(model="llama2")

# Define conversation memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Define a prompt template to enhance chatbot responses
prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template=""" 
    You are an AI assistant. You have a memory and should respond in a conversational manner.

    Chat history:
    {chat_history}

    User: {user_input}
    AI:
    """
)

# Create LangChain's conversational chain
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Initialize chat history in Streamlit session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for idx, msg in enumerate(st.session_state.messages):
    # Ensure unique key for each message
    unique_key = f"{msg['key']}_{idx}_{uuid.uuid4()}"
    message(msg["content"], is_user=msg["is_user"], key=unique_key)

# User input form
with st.form(key='chat_form'):
    user_input = st.text_input("Type your message...", key="input")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    # Get AI response
    response = chain.run(user_input)
    
    # Generate unique key for the user message
    user_key = f"user_{uuid.uuid4()}"
    # Store user message with unique key
    st.session_state.messages.append({"content": user_input, "is_user": True, "key": user_key})
    
    # Generate unique key for the AI response
    ai_key = f"ai_{uuid.uuid4()}"
    # Store AI response with unique key
    st.session_state.messages.append({"content": response, "is_user": False, "key": ai_key})
    
    # Clear the input box by rerunning the script
    st.rerun()