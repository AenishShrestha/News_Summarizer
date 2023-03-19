import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.llms import OpenAIChat
from langchain.chains.question_answering import load_qa_chain
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_pills import pills
import datetime
import streamlit.components.v1 as components
import streamlit_analytics
import markdown

with streamlit_analytics.track():
    # create two equal-width columns
    left_column, right_column = st.columns(2)

    # add the header to the second column
    with right_column:
        st.header("News GPT")



    BADGES = """
    <a href="https://github.com/AenishShrestha/News_Summarizer" title="Star Repo" target="_blank"><img src="https://img.shields.io/github/stars/lukasmasuch/streamlit-pydantic.svg?logo=github&style=social"></a>
    <a href="https://twitter.com/aenish_shrestha" title="Follow on Twitter" target="_blank"><img src="https://img.shields.io/twitter/follow/lukasmasuch.svg?style=social&label=Follow"></a>
    """
    st.markdown(BADGES,unsafe_allow_html=True)

    current_time = datetime.datetime.now()

    st.subheader("Select One News Category Website: ")
    news = pills("", ["Nepal | Current","Nepal | National","International", "Sports"], ["❓", "📱","🔎","⚽"])

    # if news == "Kantipur":
    #      url='https://ekantipur.com/'

    # if news == "National":
    #      url='https://www.hamropatro.com/news/national'

    # if news == "International":
    #      url = 'https://www.hamropatro.com/news/international'

    # if news == "Sports":
    #     url = 'https://www.hamropatro.com/news/sports'


    url_mapping = {
        "Nepal | Current" : "https://www.hamropatro.com/news",
        "Nepal | National": "https://www.hamropatro.com/news/national",
        "International": "https://www.hamropatro.com/news/international",
        "Sports": "https://www.hamropatro.com/news/sports"
    }

    url = url_mapping[news]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"
    }
    
    try:
        response = requests.get(url, headers=headers)
        html = response.content

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()


    
        # Convert the text content to markdown
        markdown_text = markdown.markdown(text)
        text = markdown_text.replace('\n','')
        text_bytes = text.encode("utf-8")
        text_splitter = RecursiveCharacterTextSplitter( 
                    chunk_size = 800,
                    chunk_overlap  = 20
                )
        texts = text_splitter.split_text(text)


        #openai 
        OPENAI_API_KEY = st.secrets["OPEN_API_KEY"])

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        prefix_messages = [
            {"role": "system", "content": "You are a helpful AI Website Explainer."},
            {"role": "user", "content": "Create long form summary. Output as a markdown."},
            {"role": "assistant", "content": "Okay what is the name of website?"},]

        query = 'Create long form summary. Only List the bulletpoints. Give the headline in h1 tag. Output as a markdown.Do not repeat yourself.'

        with open("foo.pkl", 'wb') as f:
                pickle.dump(embeddings, f)

        with open("foo.pkl", 'rb') as f:
            new_docsearch = pickle.load(f)

        docsearch = FAISS.from_texts(texts, new_docsearch)

        docs = docsearch.similarity_search(query)

        chain = load_qa_chain(OpenAIChat(openai_api_key=OPENAI_API_KEY,temperature=0,prefix_messages=prefix_messages), chain_type="stuff")

        result = chain.run(input_documents=docs, question=query)
        result = result.replace("\n\n--","").replace("\n--","").strip()

        # Convert to desired format
        time_in_desired_format = current_time.strftime("%d %b %Y, %H:%M:%S")

        # Display current date and time in desired format
        st.write('Current Date & Time:', time_in_desired_format)
        st.markdown(result)
        st.markdown("Made with ❤️ by [Aenish Shrestha](https://twitter.com/aenish_shrestha)")
        components.iframe("https://aenishshrestha.substack.com/embed",height=500)
    except Exception as e :
        st.error(f"Error : {str(e)}")
