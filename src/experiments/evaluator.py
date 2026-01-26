"""
Experiment Evaluator (LLM-as-a-Judge)
Scores the extracted descriptions based on their utility for answering user queries.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentEvaluator:
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-09-2025"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def generate_test_query(self, image_name: str) -> str:
        """
        Generate a plausible user query for the image based on its name/context.
        (In a real scenario, this would use ground truth or look at the image itself).
        For this experiment, we'll ask Gemini to guess a query based on the filename 
        or simply use a generic robust one if the filename is opaque.
        """
        # Simple heuristic based on filename components
        clean_name = image_name.replace("_", " ").replace(".png", "")
        return f"How do I use the feature shown in {clean_name}?"

    def evaluate_description(self, description: str, query: str, strategy_name: str) -> Dict:
        """
        Ask the Judge to score the description.
        """
        judge_prompt = f"""
You are an unbiased evaluator (LLM-as-a-Judge).
Your task is to rate the quality of an Image Description for a RAG system.

SCENARIO:
- A user asks: "{query}"
- The RAG system retrieves this text description of an image:
"{description}"

EVALUATION CRITERIA:
1. Helpfulness (1-5): Does this description contain enough info to answer the question?
2. Specificity (1-5): Is it specific to the image or generic fluff?
3. Actionability (1-5): Can the user take action based on this?

OUTPUT JSON ONLY:
{{
    "helpfulness_score": int,
    "specificity_score": int,
    "actionability_score": int,
    "reasoning": "string"
}}
"""
        try:
            response = self.model.generate_content(judge_prompt)
            # clean response text to ensure json
            text = response.text.replace("```json", "").replace("```", "").strip()
            score_data = json.loads(text)
            return score_data
        except Exception as e:
            logger.error(f"Evaluation failed for {strategy_name}: {e}")
            return {"error": str(e)}

    def run_evaluation(self, run_dir: Path):
        """
        Evaluate a specific experiment run directory.
        """
        summary_path = run_dir / "experiment_summary.json"
        if not summary_path.exists():
            logger.error(f"No summary found at {summary_path}")
            return

        with open(summary_path, "r", encoding="utf-8") as f:
            experiment_data = json.load(f)
            
        evaluation_results = []
        
        for item in experiment_data:
            image_name = item["image_name"]
            strategies = item["strategies"]
            
            # Generate a consistent query for this image
            query = self.generate_test_query(image_name)
            logger.info(f"Evaluating image: {image_name} | Query: {query}")
            
            item_scores = {
                "image_name": image_name,
                "query": query,
                "scores": {}
            }
            
            for strat_name, strat_data in strategies.items():
                if strat_data["status"] == "success":
                    # Load actual output text
                    out_file = strat_data.get("output_file")
                    if out_file and os.path.exists(out_file):
                        with open(out_file, "r", encoding="utf-8") as f:
                            description = f.read()
                            
                        score = self.evaluate_description(description, query, strat_name)
                        item_scores["scores"][strat_name] = score
                        print(f"  - {strat_name}: Helpfulness={score.get('helpfulness_score', 0)}")
                    
            evaluation_results.append(item_scores)
            
        # Save evaluation report
        report_path = run_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2)
            
        self.generate_markdown_report(evaluation_results, run_dir)
        
    def generate_markdown_report(self, results: List[Dict], run_dir: Path):
        """Generates a human-readable markdown report."""
        md = "# Extraction Experiment Evaluation Report\n\n"
        
        # Calculate averages
        avgs = {}
        for item in results:
            for strat, score in item["scores"].items():
                if "error" in score: continue
                if strat not in avgs: avgs[strat] = {"h": 0, "s": 0, "a": 0, "count": 0}
                avgs[strat]["h"] += score.get("helpfulness_score", 0)
                avgs[strat]["s"] += score.get("specificity_score", 0)
                avgs[strat]["a"] += score.get("actionability_score", 0)
                avgs[strat]["count"] += 1
                
        md += "## Summary of Averages (Scale 1-5)\n\n"
        md += "| Strategy | Helpfulness | Specificity | Actionability |\n"
        md += "|----------|-------------|-------------|---------------|\n"
        
        for strat, data in avgs.items():
            if data["count"] > 0:
                h = data["h"] / data["count"]
                s = data["s"] / data["count"]
                a = data["a"] / data["count"]
                md += f"| {strat} | {h:.1f} | {s:.1f} | {a:.1f} |\n"
                
        md += "\n## Detailed Results\n\n"
        for item in results:
            md += f"### Image: {item['image_name']}\n"
            md += f"**Query**: *{item['query']}*\n\n"
            for strat, score in item["scores"].items():
                if "error" in score:
                    md += f"- **{strat}**: Error evaluating\n"
                else:
                    md += f"- **{strat}**: H={score.get('helpfulness_score')} | S={score.get('specificity_score')} | A={score.get('actionability_score')}\n"
                    md += f"  - *Reasoning*: {score.get('reasoning')}\n"
            md += "\n---\n"
            
        with open(run_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(md)
        logger.info(f"Markdown report generated at {run_dir / 'report.md'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Directory of the experiment run to evaluate")
    args = parser.parse_args()
    
    evaluator = ExperimentEvaluator()
    evaluator.run_evaluation(Path(args.run_dir))
