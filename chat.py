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
    web_path = "https://internshala.com/internship/details/work-from-home-data-analytics-internship-at-native-engineering1735614729"
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
# print(res.content)


# Text in str convert into json
from langchain_core.output_parsers import JsonOutputParser
json_parser = JsonOutputParser()
json_res = json_parser.parse(res.content)
# print(json_res)
# print(type(json_res))


# Dataset
import pandas as pd
df=pd.read_csv("my_portfolio.csv")
# print(df.head(2))

# Function to find links that contain certain keywords using regex
def find_matching_links(job_skills, n_results=2):
    job_skills = "".join(job_skills).lower()
    print(job_skills)
    print("*"*20)
    matching_rows = []
    
    for _, row in df.iterrows():
        # Convert techstack to lowercase for case-insensitive matching
        techstack = row["Techstack"].lower()
        # for keyword in job_skills.split(" "):
        #     print(keyword)
        # Check if any job skill keyword matches the tech stack
        if any(keyword in techstack for keyword in job_skills.split(" ")):
            matching_rows.append(row)
    
    # Return the top n matching links
    return [row["Links"] for row in matching_rows[:n_results]]

# Example query with job skills
job_skills = ['Strong Computer Science fundamentals',
  'Programming experience with at least one modern language such as Java, Python, Golang',
  'Experience with Infrastructure as Code / DevOps practices and technologies like Terraform',
  'Experience with Cloud services (e.g. AWS Cloud EC2, S3, DynamoDB, Azure Cloud, etc.)',
  'Experience developing and using microservices']
matching_links = find_matching_links(job_skills)
# print(matching_links)


job = json_res
# print(job)
# print(job['skills'])
# print(type(job['skills']))


prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Parker, a Chief development executive at Parker Industries. ParkEr is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools. 
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
        process optimization, cost reduction, and heightened overall efficiency. 
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of ParkEr 
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase Parker Industries's portfolio: {link_list}
        Remember you are Parker, CEO at Parker Industries. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )

chain_email = prompt_email | llm
res = chain_email.invoke({"job_description": str(job), "link_list": find_matching_links(job['skills'])})
# print(find_matching_links(job['skills']))
print(res.content)
