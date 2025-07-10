from typing import Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.prompt.manus_backup import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
import openai  # Add OpenAI import for GPT call
import os
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import logging


logger = logging.getLogger(__name__)

class Manus(ToolCallAgent):
    """A versatile general-purpose agent."""

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 5

    # Add general-purpose tools to the tool collection
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
#Optional
    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        result = await super().think()

        self.next_step_prompt = original_prompt

        return result

    async def run(self, prompt: str) -> str:
        """Run Manus agent, then extract exactly the next action using fixed prefixes."""
        # 1) Let Manus think / gather tools / dialogue as before
        full_prompt = self.system_prompt + "\n\n" + prompt
        logger.info("\n========== PROMPT SENT TO OpenManus ==========\n%s\n", full_prompt)
        raw_response = await super().run(full_prompt)
        logger.info("\n========== RAW RESPONSE FROM OpenManus ==========\n%s\n", raw_response)

        llama_instruction = (
            "You are a medical assistant helping extract the doctor's next intended actions, output only one of the followings, with no extra text, explanations, or formatting: \n\n"
            "1) One or more test requests, each on its own line and prefixed **REQUEST TEST:** (a list of **all** tests the doctor explicitly asked for using “REQUEST TEST: …”) \n"
            "   e.g. `REQUEST TEST: Chest X‑Ray`\n\n"
            "2)  Follow‑up question (no prefix), e.g. `Have you had any fevers?` (the follow‑up questions the doctor intends to ask next) \n\n"
            "3) A final diagnosis, prefixed **DIAGNOSIS READY:** (only if the doctor explicitly stated “DIAGNOSIS READY: …”  **and** the diagnosis is based on real (not assumed) data.)\n"
            "   e.g. `DIAGNOSIS READY: Pneumonia`\n\n"
            "**Rules:**\n"
    "  1. Never simulate or assume any test results, clinical findings, or facts; you can only report what was literally stated.\n"
    "  2. If the doctor’s text contains “DIAGNOSIS READY” **but** that diagnosis is based on assumed or imaginary results, omit the diagnosis.\n"
    "  3. If multiple `REQUEST TEST:` exist, list them all (each on its own line). \n"
                )

        messages = [
            {"role": "system", "content": llama_instruction},
            {"role": "user", "content": raw_response}
        ]
        logger.info("\n========== LLaMA4 INPUT MESSAGES ==========\n%s\n", messages)

        # 3) Call LLaMA4 via OpenAI-compatible API
        try:
            # Configure OpenAI client
            client = openai.OpenAI(
                api_key=" ",  # Your API key
                base_url=" "  # Your API endpoint
            )
            resp = client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",  # Your LLaMA4 model name
                messages=messages,
                temperature=0.0,
                max_tokens=100
            )
            result = resp.choices[0].message.content.strip()
            logger.info("\n========== LLaMA4 RAW OUTPUT ==========\n%s\n", result)

        except Exception as e:
            logger.error(f"Error during LLaMA4 API call: {str(e)}")
            return "Error processing LLaMA4 output."

        logger.info("\n========== FINAL FORMATTED OUTPUT ==========\n%s\n", result)
        return result

    async def cleanup(self):
        """Clean up Manus agent resources."""
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()