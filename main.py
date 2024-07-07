from langchain_anthropic import ChatAnthropic

import getpass
import os

os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)

ai_msg = llm.invoke('hi there')

print(ai_msg.content)
