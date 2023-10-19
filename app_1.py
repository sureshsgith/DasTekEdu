from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
import os
os.environ["OPENAI_API_KEY"]="sk-POyh72"+"mWbhIMJ8QATu63T3Bl"+"bkFJYyk2492ZRqgPSE"+"uG4FaK"
os.environ["SERPAPI_API_KEY"]="beb8949037"+"697a72c43ed791a"+"53a83e7507e9f8a84"+"28caa923eeae8ca37929fe"
import streamlit as st
import time
import easyocr
import numpy as np
from PIL import Image
st.markdown("<h1 align=center>DastekEdu📖</h1>",unsafe_allow_html=True)
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
# Define which tools the agent can use to answer user queries
search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events.if don't know about current events and you know about data upto 2021.you should use this tool.For explanation you already have the data so should don't use this tool for that"
    ),
]
# Set up the base template
template = """Your name is DastekEdu. I want to act as a great Educator in all field. you task is to give the answer with clear explanations step by step and don't give direct answers. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take,should be {tool_names}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:



Begin!

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

def img_txt():
    # Load EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Read text from the uploaded image
    image = Image.open(uploaded_image)
    image_np = np.array(image)
    detected_text = reader.readtext(image_np)


    # Extract the detected text
    extracted_text = ""
    for detection in detected_text:
        text = detection[1]
        extracted_text += text + " "
    return extracted_text

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Try to parse out the action and action input using the regex
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if match:
            action = match.group(1).strip()
            action_input = match.group(2).strip(" ").strip('"')
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)
        
        # If the output doesn't match the action format, treat it as an observation
        if "\nObservation:" in llm_output:
            observation = llm_output.split("\nObservation:", 1)[-1].strip()
            return AgentAction(tool="Observation", tool_input="", log=llm_output)
        
        # If none of the above conditions are met, treat the output as a finish
        return AgentFinish(
            return_values={"output": llm_output.strip()},
            log=llm_output,
        )

output_parser = CustomOutputParser()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

if "messages" not in st.session_state:
        st.session_state.messages = []

for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
if uploaded_image is not None:
    prompt=img_txt()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        ai_response=agent_executor.run(prompt)
        # for chunk in ai_response.split():
        #     full_response += chunk + " "
        #     time.sleep(0.05)
        #     message_placeholder.write(full_response + "▌")
        message_placeholder.write(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
if prompt := st.chat_input("Ask a Question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        ai_response=agent_executor.run(prompt)
        # for chunk in ai_response.split():
        #     full_response += chunk + " "
        #     time.sleep(0.05)
        #     message_placeholder.write(full_response + "▌")
        message_placeholder.write(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
