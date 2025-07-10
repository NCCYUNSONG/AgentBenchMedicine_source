import logging
from openai import OpenAI
from app.tool.base import BaseTool, ToolResult

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_IMAGE_ANALYSIS_DESCRIPTION = """A tool for analyzing images. Always need an image URL."""

# """A tool for analyzing medical images , with context from the OpenManus doctor agent. Provide an image URL, and dialogue history to guide the analysis."""


class ImageAnalysisTool(BaseTool):
    name: str = "image_analysis"
    description: str = _IMAGE_ANALYSIS_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "The URL of the image to analyze",
            },
            "question": {
                "type": "string",
                "description": "The question to be answered.",
            },
        },
        "required": ["image_url"],
    }

    async def execute(
        self,
        image_url: str,
        question: str = "",
        **kwargs,
    ) -> ToolResult:
        """
        Execute the image analysis tool using LLaMA-4 with Manus doctor agent context.

        Returns:
            ToolResult with the analysis output or error.
        """
        logger.info("🔧 ImageAnalysisTool executing")
        if not image_url:
            return ToolResult(error="image_url is required for 'image_analysis' action")
        # if not dialogue_history:
        #     return ToolResult(error="dialogue_history is required for 'image_analysis' action")

        # Hardcoded LLaMA-4 configuration
        client = OpenAI(
            api_key="sk-KOWcpxjdDUvC7Yig_JZPlw",  # Replace with actual API key
            base_url="http://pluto/v1/"
        )
        model = "Llama-4-Maverick-17B-128E-Instruct-FP8"

        # Build system prompt with default Manus context
        prompt = f"Question:\n{question}"
        prompt = "Based on the question, analyze the image and provide a description."

        # Log the prompt
        # logger.info("ImageAnalysisTool prompt:\n%s", prompt)

        # Build multimodal message
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the image and provide a description."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        try:
            # Call LLaMA-4 for image analysis
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512
            )
            analysis_result = resp.choices[0].message.content.strip()
            return ToolResult(output=f"Image analysis result: {analysis_result}")

        except Exception as e:
            return ToolResult(error=f"Image analysis failed: {str(e)}")

    async def cleanup(self):
        """Clean up resources (none required)."""
        pass