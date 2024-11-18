'''
This code is for invoking answers based on multiple choice question format queries.
'''

import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from get_embedding_function import get_embedding_function
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']

CHROMA_PATH = "/home/jiyoon/UpstageNLP_Team8/rag_upstage/chroma"
TEST_PATH = "/home/jiyoon/UpstageNLP_Team8/rag_upstage/test_data/test_samples.csv"

PROMPT_TEMPLATE = """
Answer the question based only on the following context
If the answer is not present in the context, please write "The information is not present in the context.":

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    prompts, answers = read_test_data(TEST_PATH)
    print(prompts)
    for prompt in prompts: 
        query_rag(prompt)
    
    

def read_test_data(data_path):
    data = pd.read_csv(data_path)
    prompts = data['prompts']
    answers = data['answers']
    return prompts, answers

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=10)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = ChatUpstage(api_key = upstage_api_key)

    responses = []
    response = model.invoke(prompt)
    responses.append(response.content)

    for i in responses:
        print(i)
        print('-'*10)

    return response


if __name__ == "__main__":
    main()
