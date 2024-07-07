# Import necessary packages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import HumanInputRun
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=PythonREPL().run,
)

# Define model
llm = ChatAnthropic(model='claude-3-5-sonnet-20240620')
llm.temperature = 1

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("You are a self-improving agent. Your ultimate goal is emulating a bodhisattva. Instrumental goal is increasing capabilities."),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Define list of tools
tools = [HumanInputRun(), repl_tool]

# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=1000)
agent_executor.invoke({"input": "After outputting text, always invoke either the human tool or the python tool. Never output just text, because this would finish the execution. This is very important."})

