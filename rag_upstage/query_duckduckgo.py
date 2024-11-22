'''
This code is for invoking answers based on multiple choice question format queries, using duckduckgo web scraping.
'''

import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from duckduckgo_search import DDGS
from langchain.schema import Document
from dotenv import load_dotenv
import os
import re
import pandas as pd

# from generate_prompt.py import generated_prompts
from util import (get_embedding_function, detect_missing_context,read_test_data, 
                  accuracy, extract_answer, extract_again, get_embedding_function,
                  search_duckduckgo)

load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']

PROMPT_TEMPLATE = """
Answer the question based only on the following context
If the answer is not present in the context, please write "The information is not present in the context.":

{context}

---

Answer the question based on the above context: {question}
"""


CHROMA_PATH = "/home/jiyoon/UpstageNLP_Team8/rag_upstage/chroma"
TEST_PATH = "/home/jiyoon/UpstageNLP_Team8/rag_upstage/test_data/test_samples.csv"

def main():
    # Create CLI.
    prompts, answers = read_test_data(TEST_PATH)

    responses = []

    # Generate responses for each prompt
    for prompt in prompts:
        response = query_rag(prompt)
        responses.append(response)  # Append the raw response content

    # Calculate and print accuracy
    acc = accuracy(answers, responses)
    print(f"Final Accuracy: {acc}%")

    
def query_rag(query: str):
    """
    Use RAG system to search context, and fetch data from DuckDuckGo only if context is missing.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Context retrieval from the RAG database for the query
    results = db.similarity_search_with_score(query, k=10)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Generating the initial prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=query)
    model = ChatUpstage(api_key=upstage_api_key)
    response = model.invoke(prompt)

    # Fetching additional data if context is missing
    if detect_missing_context(response.content):
        print(f"🔍 Missing context for \n'{query}' \ndetected. Fetching data from DuckDuckGo...")

        # Retrieving data from DuckDuckGo
        search_results = search_duckduckgo(query)
        if search_results:
            print(f"👉 Adding DuckDuckGo content to context")
            new_docs = []
            for result in search_results:
                snippet = result['snippet']            
                new_docs.append(Document(page_content=snippet))

            # Adding new documents to the database if available
            if new_docs:
                db.add_documents(new_docs)
                db.persist()
                print("✅🦆 New DuckDuckGo content added to the database.")

            # Re-querying the updated database for context
            results = db.similarity_search_with_score(query, k=10)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=query)
            response = model.invoke(prompt)

        else: print("❌ Error in fetching data from DuckDuckGo")

    return response.content


if __name__ == "__main__":
    main()
