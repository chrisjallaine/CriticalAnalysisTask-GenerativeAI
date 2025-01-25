import streamlit as st
from langchain.llms import OpenAI  # Correct import for OpenAI from langchain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory)

# STEP 1: Set up session state if not already present
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'API_Key' not in st.session_state:
    st.session_state['API_Key'] = ''

# STEP 2: Set the page configuration and display the header
st.set_page_config(page_title="Chat GPT Clone", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>How can I assist you? </h1>", unsafe_allow_html=True)

# STEP 3: Sidebar input for the API key
st.sidebar.title(":D")
st.session_state['API_Key'] = st.sidebar.text_input("What's your API key?", type="password")

# STEP 4: Button to summarize the conversation
summarise_button = st.sidebar.button("Summarise the conversation", key="summarise")
if summarise_button:
    summarise_placeholder = st.sidebar.write("Nice chatting with you my friend <3️:\n\n" + st.session_state['conversation'].memory.buffer)

# STEP 5: Define the response function using langchain
def getresponse(userInput, api_key):
    if st.session_state['conversation'] is None:
        llm = OpenAI(
            temperature=0,
            openai_api_key=api_key,
            model_name='gpt-3.5-turbo-instruct'
        )

        st.session_state['conversation'] = ConversationChain(
            llm=llm,
            verbose=True,
            memory=ConversationSummaryMemory(llm=llm)
        )

    response = st.session_state['conversation'].predict(input=userInput)
    return response

# STEP 6: Display the chat interface
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your question goes here:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        if submit_button:
            # Append user input and model response to messages
            st.session_state['messages'].append({"role": "user", "text": user_input})
            model_response = getresponse(user_input, st.session_state['API_Key'])
            st.session_state['messages'].append({"role": "ai", "text": model_response})

            # Display the conversation messages
            with response_container:
                for message in st.session_state['messages']:
                    if message["role"] == "user":
                        st.markdown(f"**User:** {message['text']}")
                    else:
                        st.markdown(f"**AI:** {message['text']}")

# STEP 7: API Key storage in session state and pass it to the model
st.session_state['API_Key'] = st.sidebar.text_input("Enter your API key", type="password")

# Ensure conversation is initialized
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None
