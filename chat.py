from langchain_groq import ChatGroq
from api import api
import getpass
import os

# Craeting LLm
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key = api
    # other params...
)

response= llm.invoke("The first person lands on moon was .. ")
# print(response.content)


#  Web Scrapper
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path = "https://jobs.nike.com/job/R-48627"
)
page_data = loader.load().pop().page_content
# print(page_data)


# prompt template
from langchain_core.prompts import PromptTemplate

# Instantiation using from_template (recommended)
prompt_extract = PromptTemplate.from_template(
    """
### Scrape text from the website:{page_data}
### INSTRUCTION:
The scrape text is from carrer's page of a  website.
Your job is to extract the job posting and return them in json format contanining
the following keys: 'role','experience', 'skills' and 'description'.
Only return the valid json.
### Valid JSON no preamble
"""
)
chain_extract = prompt_extract | llm
res = chain_extract.invoke(input={'page_data':page_data})
print(res.content)


# Text in str convert into json
from langchain_core.output_parsers import JsonOutputParser
json_parser = JsonOutputParser()
json_res = json_parser(res.content)

