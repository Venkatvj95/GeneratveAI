import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()
import os


## used the inbuild tool of wikipedia
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper= api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="search")

## side bar settings

# st.sidebar.title("Settings")
# api_key = st.text_input("Enter your Groq API key", type="password")
api_key = os.getenv("GROQ_API_KEY")

st.title("Langchain with search ")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hi, i am a chatbot who can search the web. How can i help you"}

    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="what is machine learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(api_key=api_key,model_name ="llama-3.1-8b-instant", streaming= True)
    tools = [search,wiki,arxiv]

    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors = True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts= False)
        responce = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant', "content":responce})
        st.write(responce)



    




















