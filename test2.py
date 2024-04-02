# Bring in deps
import os 
import streamlit as st 
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper 

st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here') 


title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Write me a youtube video title {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='write me a youtube video script based on this TITLE: {title}'
)

title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q4_0.gguf",
     n_ctx=2048,
    verbose=False,
    )

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

if prompt: 
    title = title_chain.run(prompt) 
    st.write(title) 
    with st.expander('Title History'): 
        st.info(title_memory.buffer)

if prompt: 
    script = script_chain.run(prompt)
    st.write(script)
    with st.expander('Script History'): 
        st.info(script_memory.buffer)
