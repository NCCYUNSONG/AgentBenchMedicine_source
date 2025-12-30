from typing import Optional
from pydantic import Field, model_validator, BaseModel
import openai
import os

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
import logging

logger = logging.getLogger(__name__)


class FinalAnswer(BaseModel):
    answer: str


class Manus(ToolCallAgent):
    """A versatile general-purpose agent."""

    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks using multiple tools"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT
    max_observe: int = 10000
    max_steps: int = 10

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), BrowserUseTool(), StrReplaceEditor(), Terminate()
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    async def think(self) -> bool:
        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:
            self.next_step_prompt = await self.browser_context_helper.format_next_step_prompt()

        result = await super().think()
        self.next_step_prompt = original_prompt
        return result

    async def cleanup(self):
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()

    async def run(self, prompt: str) -> FinalAnswer:
        """Run Manus, then extract a single‐letter QA answer as structured output."""
        full_prompt = self.system_prompt + "\n\n" + prompt
        _ = await super().run(full_prompt)  # multi-step execution

        # Gather all assistant steps
        steps = [
            msg.content.strip()
            for msg in self.memory.messages
            if msg.role == "assistant" and isinstance(msg.content, str)
        ]
        transcript = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))

        # Construct extraction prompt
        gpt_instruction = (
            "You are a QA decision assistant.\n"
            "Given a multiple-choice question and a transcript of reasoning steps, "
            "extract and return the final answer.\n"
            "The answer must be a single capital letter: A, B, C, or D.\n"
            "Do not return explanations or extra content."
        )
        messages = [
            {"role": "system", "content": gpt_instruction},
            {"role": "user", "content": f"Original Question:\n{prompt.strip()}\n\nTranscript:\n{transcript}"}
        ]
        # Log what will be passed to GPT
        logger.info("\n========== FINAL PARSING PROMPT TO GPT ==========\n%s\n", messages)

        # Call GPT
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=FinalAnswer
        )

        parsed: FinalAnswer = completion.choices[0].message.parsed
        logger.info("\n========== FINAL PARSED ANSWER ==========\n%s\n", parsed)
        return parsed
