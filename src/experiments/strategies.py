"""
Extraction Strategies Module
Defines different prompt engineering strategies for extracting meaning from images.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import google.generativeai as genai
import os
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractionStrategy(ABC):
    """Abstract base class for image extraction strategies."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-09-2025"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy."""
        pass
        
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining the persona and goal."""
        pass
        
    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """Specific instructions for the image analysis."""
        pass

    def generate(self, image_path: str, image_data: bytes, mime_type: str) -> Dict[str, Any]:
        """
        Generate description for the image using the strategy.
        """
        try:
            logger.info(f"Running strategy '{self.name}' on {os.path.basename(image_path)}")
            
            response = self.model.generate_content([
                {
                    "text": self.system_prompt + "\n\n" + self.prompt_template
                },
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_data
                    }
                }
            ])
            
            return {
                "strategy": self.name,
                "image_path": image_path,
                "output": response.text,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Strategy '{self.name}' failed: {e}")
            return {
                "strategy": self.name,
                "image_path": image_path,
                "error": str(e),
                "status": "failed"
            }


class VisualStrategy(ExtractionStrategy):
    """
    Strategy A: Visual Descriptive (Baseline)
    Focuses on describing the visual elements, layout, and appearance.
    """
    @property
    def name(self) -> str:
        return "visual_descriptive"
        
    @property
    def system_prompt(self) -> str:
        return """You are an expert technical documentation analyst. 
Your goal is to provide a comprehensive VISUAL description of the image provided.
Focus on WHAT IS SEEN: layout, text, colors, icons, and spatial organization."""

    @property
    def prompt_template(self) -> str:
        return """Please provide a detailed visual description of this image.
Include:
1. All visible text.
2. Description of icons, symbols, and graphics.
3. Layout and arrangement of elements.
4. Visual style (screenshot, diagram, table, etc.).
"""


class FunctionalStrategy(ExtractionStrategy):
    """
    Strategy B: Functional/Problem-Solution
    Focuses on the user's goal, actions, and the problems the image solves.
    """
    @property
    def name(self) -> str:
        return "functional_goal_oriented"
        
    @property
    def system_prompt(self) -> str:
        return """You are a Technical Support Expert.
Your goal is to interpret this image from a USER'S perspective. 
Focus on WHAT IT DOES: actionable steps, problem-solving, and functional context.
Ignore purely decorative elements."""

    @property
    def prompt_template(self) -> str:
        return """Analyze this image largely ignoring visual style. Answer:
1. What is the PURPOSE of this screen/diagram?
2. What ACTION should the user take?
3. What PROBLEM does this help solve?
4. Explain any error codes or status indicators in terms of user troubleshooting.
"""


class JsonStrategy(ExtractionStrategy):
    """
    Strategy C: Structured Data (JSON)
    Focuses on extracting key-value pairs for machine processing.
    """
    @property
    def name(self) -> str:
        return "structured_json"
        
    @property
    def system_prompt(self) -> str:
        return """You are a Data Extraction Bot.
Your goal is to parse the image content into a strict JSON format.
Output ONLY valid JSON. Do not include markdown formatting or explanations."""

    @property
    def prompt_template(self) -> str:
        return """Extract the following fields from the image in JSON format:
{
    "type": "screenshot | diagram | icon | other",
    "title": "Main title or heading found",
    "key_text": ["List of critical text labels"],
    "actionable_elements": ["List of buttons or inputs"],
    "status_indicators": [{"name": "indicator name", "state": "value"}]
}
If a field is not present, use null or empty list.
"""
