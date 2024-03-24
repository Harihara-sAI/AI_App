import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import os

os.environ["OPENAI_API_KEY"] = "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"
 
# Sidebar contents
with st.sidebar:
    st.title('App')
    st.write("This application allows you to upload one or more financial reports in PDF format. You can ask questions related to the PDF files and this application will give you answers.")
    
 
load_dotenv()

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text
 
def main():
    st.header("Financial Reports Analyzer")
 
 
    # upload a PDF file
    pdf_docs = st.file_uploader("Upload your PDF",type='pdf',accept_multiple_files=True)
 
    # st.write(pdf)
    if pdf_docs:
        text=get_pdf_text(pdf_docs)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        #store_name = pdf.name[:-4]
        #st.write(f'{store_name}')
        # st.write(chunks)
 
        #if os.path.exists(f"{store_name}.pkl"):
        #    with open(f"{store_name}.pkl", "rb") as f:
        #        VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        #else:
        #    embeddings = OpenAIEmbeddings()
        #    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        #    with open(f"{store_name}.pkl", "wb") as f:
        #        pickle.dump(VectorStore, f)
 
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions here:")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()