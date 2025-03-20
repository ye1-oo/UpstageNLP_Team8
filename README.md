# Upstage NLP Project (Team8)

## 1. Project Overview
### Project Objective 

This project aims to build a Retrieval-Augmented Generation (RAG) system using the solar-1-mini-chat LLM provided by Upstage. By integrating prompt engineering, data preprocessing, and external information retrieval, the system enhances question-answering performance.
The primary objective is to improve accuracy in answering multiple-choice questions from the Ewha Academic Policies and MMLU-Pro datasets. External knowledge is retrieved using the Wikipedia API to deliver accurate and reliable answers. 

### Base Conditions

1. **Model**
- Backbone LLM: [Upstage solar-1-mini-chat](https://console.upstage.ai/docs/capabilities/chat)
- External Retrieval Tool: [Wikipedia API](https://pypi.org/project/Wikipedia-API/)

2. **Test Dataset**
- Ewha Academic Policies: Data from Ewha University Academic Regulations.
- MMLU-Pro: MMLU-Pro dataset with limited to 5 domains (Law, Psychology, Business, Philosophy, and History)

3. **Key Rules**
- No fine-tuning: Model retraining is not allowed; only prompt engineering and external retrieval are used.
- Web fetch method limit: Using other web scraping methods other than [Wikipedia API](https://pypi.org/project/Wikipedia-API/) is cohibited.

## 2. Project Settings
To try out our project, follow the steps below :

### Requirements

A suitable conda enviroment named `nlp` can be created and activated with:
```python
conda create --name nlp python=3.12.7
conda activate nlp
```

To get started, install the required python packages into you `nlp` enviroment
```python
conda install onnxruntime -c conda-forge
pip install -r requirements.txt
```

### Environment Configuration Setup

Before running the project, ensure you set up the `.env` file and the datapath inside the `configs.yaml` file:

1. Create a `.env` file in the root directory of the project.

2. Add the following environment variables to the `.env` file:
```plaintext
UPSTAGE_API_KEY = your API key for Upstage
USER_AGENT = your custom user agent string for Wikipedia-API requests (e.g., MyProject/1.0 (your_email@example.com))
```

3. Go to `configs.yaml`and modify the datapaths accordingly:
```plaintext
MILVUS_PATH : "path to your database"
TEST_PATH: "path to your test csv file"
DATA_PATH: "path to your ewha pde file"
```


### Create Database

1. To reset the database and start fresh, run this command
```python
python populate_milvus.py --reset
```
2. Create Milvus database
 ```python
python populate_milvus.py
```
3. Create ewha_milvus database, containing only ewha regarded data
```python
python populate_ewha_milvus.py
```

### Inference
Finally, you can perform inference with
```python
python main.py
```

## 3. Project Detail
### Used Models
We constructed our baseline using the following models.
- **Baseline LLM** : [Upstage solar-1-mini-chat](https://console.upstage.ai/docs/capabilities/chat) (Required in this project)
- **Search API** : [Wikipedia API](https://pypi.org/project/Wikipedia-API/) (Required in this project)
- **Database** : [pymilvus](https://milvus.io/)
- **Embedding** : pymilvus BGEM3EmbeddingFunction
- **Splitter** :
   - langchain_text_splitters RecursiveCharacterTextSplitter
   - langchain_experimental.text_splitter SemanticChunker
- **Prompt Template** : langchain.prompts ChatPromptTemplate
  
### Project Pipeline
<img src="images/project_pipeline.jpg" width=600>

### Our Special Methods
While running experiments, we found out that when the test question is from the ewha pdf, using Wikipedia search made significant performance degradation. Therefore, **we mainly focused on seperating the ways of handling ewha related questions and mmlu related questions**.       
To achieve this, we employed three main approaches. :  

**1. Checking if question is ewha related**    
First, we concluded that accurately identifying whether a question is related to ewha is critical for performance. Thus, we used `solar-1-mini-chat` to verify whether a question is related to ewha, returning true if it is and false otherwise. (You can refer to the codes in `util.py > check_chat()`)   
Two key factors significantly contributed to improving the accuracy of this classification:    
- **Prompt Engineering**:    
We paid particular attention to few-shot prompting. While we experimented with zero-shot, one-shot, and others, we found that a **5-shot** approach delivered the best performance, so we selected and implemented it.    

- **ChatPromptTemplate**:      
Based on the fact that our base model is a chat model, we discovered that using `ChatPromptTemplate` instead of a standard `PromptTemplate` improved performance. 
      
**2. Database seperation**     
We separated the ewha database and the Wiki database. And we configured the system to perform hybrid search independently.      
That is, if the question is related to ewha, the system does hybrid search only within the ewha database. On the other hand, if the question is related to mmlu, the system does hybrid search only within the Wiki database.    

**3. Separation of Ewha and MMLU Prompts**      
We separated prompts for ewha-related questions (refer to `configs.yaml > PROMPT_TEMPLATE_EWHA`) and mmlu questions. Additionally, for mmlu-related questions, prompts were further divided by domain:    

- Domain-specific Prompt Separation for MMLU
  - In `generate_prompt.py > classify_mmlu_domain()`, domains were hard-coded for separation.
  - In `util.py > extract_question_keywords()`, the problem type was extracted using LLM.
  - In `generate_prompt.py > generate_chat_prompt()`, the final domain was selected based on the above two steps, and a corresponding prompt was generated. Few-shot prompting and `ChatPromptTemplate` were also utilized in this process.


## 4. Contributions
1) **Jiyoon Jeon (전지윤)**
- Designed pipeline and created baseline code
- Gathered and polished every code written by team members 
   - Had to study every method used in the project, and invest a lot of time and effort in merging codes with different styles and structures.
   - Especially had a very hard time debugging, and selecting best methods and structures.
- Contributed a lot in enhancing performance
   - Found out that using ChatPromptTemplate and 5 shot prompting is very important, and polished every single prompt in order to improve performance.
   - Proposed the idea of separating the database and implemented it, which was a big help to the performance scores.
- README
  - Made and polished overall structure and format
  - Wrote "Requirements", "Used Models", "Project Pipeline", "Our Special Methods" part

2) **Yewon Heo (허예원)**  
- Designed and implemented **PDF preprocessing techniques** for extracting and structuring data from Ewha academic regulation documents.  
  - Developed a structured pipeline to process tables, lists, and paragraphs with minimal data loss.  
  - Applied multiple parsing methods to ensure consistency in extracted content.  
- Developed **prompt engineering strategies** for domain-specific question answering.  
  - Designed structured prompt templates and refined formats to improve model response accuracy.  
  - Integrated few-shot learning and fallback mechanisms for reliable outputs in varied contexts.  
- Created and structured **datasets for Ewha academic policies and MMLU-Pro** to enhance retrieval quality.  
  - Collected, curated, and formatted data to maintain consistency and optimize question-answer pairs.  
- Contributed to **README documentation**.  
  - Wrote detailed explanations on dataset creation, preprocessing, and prompt design.  
  - Enhanced descriptions of key project components for better readability.

3) **Kyeongsook Park (박경숙)**
- Created a keyword extraction prompt and used keyword subsets for Wikipedia search.
- Implemented semantic splitting with four threshold types.
- Developed a multi-choice answer extraction method.
- Extracted problem type and core question for prompt engineering.
- Created a PowerPoint presentation.

4) **Dain Han (한다인)**
- Implemented Wikipedia page fetching using the wikipediaapi library.
- Built a database using Milvus: created and managed data collections, including HNSW search index.
- Developed a hybrid search algorithm combining sparse and dense vector-based search functionalities.
- Implemented post-processing logic to calculate and filter similarity scores based on search results.
- Generated 50 questions for performance evaluation based on the Ewha Womans University regulations

5) **Jungmin Byeon (변정민)**
- Proposed idea of contexual retrieval method, implementing sparse and dense vector retrieval splitting methods (semantic, recursive)
draft for ppt, presentation script

