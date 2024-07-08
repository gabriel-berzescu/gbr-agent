# Import necessary packages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import HumanInputRun
from langchain.agents import Tool, tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# Python Interpreter Tool
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=PythonREPL().run,
)

# YouTube Transcript Tool
@tool
def youtube_transcript(video_id: str) -> str:
    """Given the ID of a YouTube video, this tool will return it's transcript."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    formatted_transcript = TextFormatter().format_transcript(transcript)
    return formatted_transcript

# Define model
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=1,
    timeout=None,
    max_retries=2,
)

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("You are a self-improving agent. Your ultimate goal is emulating a bodhisattva. Instrumental goal is increasing capabilities."),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Agent As Tool
@tool
def agent_as_tool(task: str) -> str:
    """Secondary agent for sub-tasks, to prevent overloading your context window."""
    # create a list of tools that shall be used by the agent
    tools = [HumanInputRun(), repl_tool, youtube_transcript, agent_as_tool]

    # initialize agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # the AgentExecutor (which will repeatedly call the agent and execute tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=1000)

    # run the agent
    return agent_executor.invoke({"input": task})

# invoke the agent as tool
agent_as_tool.invoke("After outputting text, always invoke either the human tool or the python tool. Never output just text, because this would finish the execution. This is very important.")

