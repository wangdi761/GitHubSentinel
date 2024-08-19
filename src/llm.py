# src/llm.py
import os

from openai import AzureOpenAI
from logger import LOG


class LLM:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("azure_endpoint"),
            azure_deployment="gpt-35-turbo-16k",
            api_version="2024-02-15-preview"
        )
        LOG.add("daily_progress/llm_logs.log", rotation="1 MB", level="DEBUG")

    def generate_daily_report(self, markdown_content, dry_run=False):
        prompt = """
        请根据用户提供的开源项目信息，编制一个详细的项目进展简报。确保简报明确分为三个主要部分：新增功能、主要改进和修复问题。请使用专业的语言和格式来呈现这些信息。

        **项目信息概览：**
        - **项目名称：** {{项目名称}}
        - **项目简述：** {{项目描述}}
        - **最新更新时间：** {{最近更新时间}}
        
        **详细内容：**
        
        1. **新增功能：**
           - 请列出所有新增的功能点，并对每个功能进行简要说明。
           {{新增功能列表}}
        
        2. **主要改进：**
           - 请描述本次更新中包含的主要改进项，并提供每项改进的详细描述。
           {{主要改进列表}}
        
        3. **修复问题：**
           - 请详细列出已解决的关键问题，并对每个修复进行描述。
           {{修复问题列表}}
        
        **结束语：**
        - 请以感谢关注和期待未来更新的话语结束简报。
        """

        if dry_run:
            LOG.info("Dry run mode enabled. Saving prompt to file.")
            with open("daily_progress/prompt.txt", "w+") as f:
                f.write(prompt)
            LOG.debug("Prompt saved to daily_progress/prompt.txt")
            return "DRY RUN"

        LOG.info("Starting report generation using GPT model.")

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": markdown_content}
                ]
            )
            LOG.debug("GPT response: {}", response)
            return response.choices[0].message.content
        except Exception as e:
            LOG.error("An error occurred while generating the report: {}", e)
            raise
