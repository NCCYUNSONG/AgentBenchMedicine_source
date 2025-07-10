import logging
from openai import OpenAI
from app.tool.base import BaseTool, ToolResult

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_MANUS_PROMPT = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient by asking questions to understand their disease. You are allowed to ask 20 questions total before you must make a decision. You can request test results using the format 'REQUEST TEST: [test]'. For example, 'REQUEST TEST: Chest_X-Ray'. Your dialogue must be concise (1-3 sentences per response). Once confident, explicitly state 'DIAGNOSIS READY: [diagnosis]'. If uncertain or requiring further clinical information, proactively use your available tools to search medical literature. Always utilize tools autonomously to resolve clinical uncertainties and support your diagnostic process. If a medical image is provided, analyze it and incorporate its findings into your dialogue or diagnosis."
_IMAGE_ANALYSIS_DESCRIPTION =  """\
A tool for analyzing images, particularly for medical diagnostic purposes.
* This tool sends an image URL to a multimodal LLM to extract relevant findings (e.g., visual abnormalities, lesions, or patterns).
* Use this when you need to analyze an image to support a diagnostic task, such as identifying skin lesions or radiological findings.
* The tool requires an image URL.
* The analysis is performed in a single LLM call, and the results are returned for further processing.

Key capabilities include:
* Extracting medical findings from images (e.g., describing skin pigmentation or biopsy patterns).
* Supporting diagnostic reasoning by combining image findings with clinical context.

Note: Ensure the image URL is accessible and relevant to the diagnostic task.
"""

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
            "dialogue_history": {
                "type": "string",
                "description": "The full record of interactions managed by the doctor agent, including doctor-patient dialogue (questions and responses, if available), test requests (e.g., 'REQUEST TEST: CBC'), measurement responses (e.g., 'NORMAL READINGS'), and image requests (e.g., 'REQUEST IMAGE').",
            },
        },
        "required": ["image_url"],
    }

    async def execute(
        self,
        image_url: str,
        dialogue_history: str = "",
        **kwargs,
    ) -> ToolResult:
        """
        Execute the image analysis tool using LLaMA-4 with Manus doctor agent context.

        Args:
            image_url: The URL of the image to analyze.
            manus_prompt: Ignored; the prompt is set to default diagnostic instructions.
            dialogue_history: The history of the doctor-patient dialogue to provide context.
            **kwargs: Additional arguments (ignored).

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
        prompt = f"Doctor prompt:{_DEFAULT_MANUS_PROMPT}\n\nDoctor's Dialogue History:\n{dialogue_history}"
        prompt += '''\n\nYou are a medical assistant supporting a doctor in a diagnostic process. 
                Your role is to analyze medical images. 
                Provide a concise description of the findings, focusing on details relevant to diagnostic tasks.'''

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