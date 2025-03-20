import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool

load_dotenv()

def main():
    print("Starting Interpreter...")


    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)


    azureLLM = AzureChatOpenAI(
        temperature=0,
        stop=["\nObservation"],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_version="2024-08-01-preview",  # Specify API version
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment="gpt-4o",
        model="gpt-4o"
    )
    tools = [PythonREPLTool()]
    agent = create_react_agent(
        prompt=prompt,
        llm=azureLLM,
        tools=tools,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke(
        input={
            "input": """generate and save in current working directory 15 QRcodes
                                that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
        }
    )
if __name__ == "__main__":
    main()