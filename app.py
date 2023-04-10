import requests
import pytz
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
    st.set_page_config(page_title="GPTNews", page_icon="🤖", layout="wide")
#     st.markdown("[![Kathmandu Technician](https://s2.gifyu.com/images/kathmandutechnician.com.gif)](https://www.kathmandutechnician.com)")
    
    sponsor = '<p style="text-align:left;"><a href="https://www.kathmandutechnician.com"><img src="https://s2.gifyu.com/images/kathmandutechnician.com.gif" alt="Kathmandu Technician" width=300 height=100%></a></p>'
    st.markdown(sponsor, unsafe_allow_html = True)

    # create two equal-width columns
    left_column, right_column = st.columns(2)

    # add the header to the second column
    with left_column:
        st.header("📰 News GPT")
        st.subheader("Stay informed, at a glance - Summarizing News With AI Powered bullet-points in real time with just one click.")



    BADGES = """
    <a href="https://github.com/AenishShrestha/News_Summarizer" title="Star Repo" target="_blank"><img src="https://img.shields.io/github/stars/lukasmasuch/streamlit-pydantic.svg?logo=github&style=social"></a>
    <a href="https://twitter.com/aenish_shrestha" title="Follow on Twitter" target="_blank"><img src="https://img.shields.io/twitter/follow/lukasmasuch.svg?style=social&label=Follow"></a>
    """
    st.markdown(BADGES,unsafe_allow_html=True)

    current_time = datetime.datetime.now(pytz.timezone('Asia/Kathmandu'))

    st.subheader("Select One News Category Website: ")
    news = pills("", ["Nepal | Current","Nepal | National","International", "Sports","AI | GPT UPDATES"], ["❓", "📱","🔎","⚽","🤖"])

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
        "Sports": "https://www.hamropatro.com/news/sports",
        "AI | GPT UPDATES":"https://www.futuretools.io/news"
    }

    url = url_mapping[news]

    

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"
    }
    
    try:
        @st.cache(persist=True, allow_output_mutation=True)
        def news():
            response = requests.get(url, headers=headers)
            html = response.content

            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text()
            with open("matches.txt", "w",encoding="utf-8") as f:
                f.write(text)



            # Convert the text content to markdown
            markdown_text = markdown.markdown(text)
            text = markdown_text.replace('\n','')
             #slicing the characters
            if url == "https://www.futuretools.io/news":
                limited_text = text[0:3000]

            else : 
                limited_text = text[500:5000]
            # limited_text = text[0:8000]
            text_bytes = limited_text.encode("utf-8")
            text_splitter = RecursiveCharacterTextSplitter( 
                        chunk_size = 800,
                        chunk_overlap  = 20
                    )
            texts = text_splitter.split_text(limited_text)
    #         st.warning(len(texts))


            #openai 
            OPENAI_API_KEY =  st.secrets["OPEN_API_KEY"]
            # Convert to desired format
            time_in_desired_format = current_time.strftime("%d %b %Y, %H:%M:%S")
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

            if url == "https://www.futuretools.io/news":
                prefix_messages = [
                    {"role": "system", "content": "You are a helpful AI Website Explainer."},
                    {"role": "user", "content": f"Create long form summary.Give Me Summarizes based on last 3 days from todays date. Notice today's date = {time_in_desired_format}.Maintain the order of the article with sorting of descending with respect to today's date. Output as a markdown.Do you understand?"},
                    {"role": "assistant", "content": "Yes i understood.I will not break the character and consider today's date to give only 3 days "},]

                query = 'Create long form summary of last 3 days from todays date : {time_in_desired_format}. List the bulletpoints. Give the headline As "LATEST AI NEWS" in h1 tag .Also notice to provide results of last 3 days. Output as a markdown.Do not repeat yourself.'

            else:
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
            # time_in_desired_format = current_time.strftime("%d %b %Y, %H:%M:%S")

            # Display current date and time in desired format
            st.write('Current Date & Time:', time_in_desired_format)
            st.markdown(result)
            return result
        news()
        st.markdown("Made with ❤️ by [Aenish Shrestha](https://twitter.com/aenish_shrestha)")
        components.iframe("https://aenishshrestha.substack.com/embed",height=500)
    except Exception as e :
        st.error(f"Error : {str(e)}")
