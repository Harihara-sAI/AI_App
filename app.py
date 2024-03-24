import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import os

#add key in the below line
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
 
    if pdf_docs:
        text=get_pdf_text(pdf_docs)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions here:")
 
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