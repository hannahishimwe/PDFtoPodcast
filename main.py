import urllib
from urllib.parse import urlparse
import json
import re
from langchain_community.vectorstores import FAISS
from app.settings import client, model, embeddings
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langchain_community.vectorstores import FAISS

min_attributes = ('scheme', 'netloc')

transcript_prompt = """
        Your task is to take the demand of the user provided to turn the context into an engaging, informative podcast dialogue. The context may be messy or unstructured, as it could come from a variety of sources like PDFs or web pages. Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points and interesting facts that could be discussed in a podcast.

        Here is the demand from the user:

        <demand>
        {demand}
        </demand>

        Here is the context:

        <context>
        {context}
        </context>

        First, carefully read through the context and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for an audio podcast.

        <scratchpad>
        Brainstorm creative ways to discuss the main topics and key points you identified in the context ensuring it specifically fulfils the user demand. Consider using analogies, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

        Keep in mind that your podcast should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

        Use your imagination to fill in any gaps in the context or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

        Write your brainstorming ideas and a rough outline for the podcast dialogue here. Be sure to note the key insights and takeaways you want to reiterate at the end.
        </scratchpad>

        Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.

        <podcast_dialogue>
        Write a 3 minute engaging, informative segment of podcast dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience. Use made-up names for the hosts and guests to create a more engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

        Format all dialogue strictly as Speaker N: Speech on separate lines, where N is the number of the speaker talking (e.g. Speaker 1 or Speaker 2). There must only be 2 speakers Use fake names in the dialogue but use only numbers for the script. Do not use bold text, asterisks, brackets, or any additional formatting. The output will be read aloud directly as audio, so keep it natural, fluent, and clear for listening.

        Make the dialogue as detailed as possible, while still staying on topic, keeping to time limit and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the context in an entertaining way.

        At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before the end.
        </podcast_dialogue>
        """


def is_valid(url, qualifying=min_attributes):
    tokens = urlparse(url)
    if all(getattr(tokens, qualifying_attr)
               for qualifying_attr in qualifying):
        return True
    else:
      raise ValueError("Invalid URL")

def analyse_demand(demand: str):
    system_prompt = """Analyse the user's request about a document.
    Identify: (1) keywords or topics, (2) any page or section limits,
    (3) type of response requested (summary, explanation, comparison, etc.)
    Return concise JSON.
    User demand: {demand}"""
    response = client.models.generate_content(
    model='gemini-2.5-flash', contents=system_prompt.format(demand=demand)
    )
    return response.text



def clean_analysis(analysis):
    match = re.search(r"```json\s*({[\s\S]*?})\s*```", analysis)
    if match:
        json_str = match.group(1)
    else:
        raise ValueError("No JSON found in the string")
    parsed = json.loads(json_str)
    return parsed 




pdf = input("Enter pdf link: ")
try:
  is_valid(pdf)
  print("Valid URL")
except ValueError:
  print("Invalid URL")


# Load and chunk contents of the blog
loader = PDFMinerLoader(pdf)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
vector_store = FAISS.from_documents(all_splits, embedding=embeddings)

demand = input("Enter demand: ")
analysis = analyse_demand(demand)
parsed = clean_analysis(analysis)

subset = all_splits
if "pages" in parsed and parsed["pages"]:
    subset = [d for d in all_splits if d.metadata.get("page") in parsed["pages"]]

# Build a temporary local FAISS index for the subset
temp_store = FAISS.from_documents(subset, embedding=embeddings)
retrieved = temp_store.similarity_search(" ".join(parsed[next(iter(parsed))]), k=5)
context = "\n\n".join([d.page_content for d in retrieved])

response = client.models.generate_content(
    model='gemini-2.5-flash', contents=transcript_prompt.format(demand=demand, context=context)
    )
transcript = response.text

clean_transcript = "\n".join(
    line for line in transcript.splitlines() #type:ignore
    if re.match(r"^\s*\*{0,2}[^:*]+:\*{0,2}\s*", line)
)

with open('transcript.txt', 'w') as output:
    output.write(clean_transcript)
