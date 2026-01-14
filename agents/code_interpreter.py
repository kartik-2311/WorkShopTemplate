"""
Code Interpreter Agent - Simple example agent for data analysis
"""

import google.generativeai as genai
from typing import Dict, Any, List
import io
import traceback
import pandas as pd
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

from .base_agent import BaseAgent, AgentResult


class CodeInterpreterAgent(BaseAgent):
    """Simple agent that analyzes CSV data using Python code"""

    def __init__(self, api_key: str):
        super().__init__(name="CodeInterpreter", api_key=api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.dataframes: Dict[str, pd.DataFrame] = {}

    def get_capabilities(self) -> List[str]:
        return [
            "Execute Python code for data analysis",
            "Load and analyze CSV files",
            "Perform statistical analysis",
        ]

    async def process(self, input_data: Dict[str, Any]) -> AgentResult:
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        files = input_data.get("files", {})

        # Load CSV files if provided
        if files:
            for filename, filepath in files.items():
                try:
                    df = pd.read_csv(filepath)
                    safe_name = (
                        filename.replace(".csv", "").replace("-", "_").replace(" ", "_")
                    )
                    self.dataframes[safe_name] = df
                    self.dataframes[filename] = df
                except Exception as e:
                    return AgentResult(
                        success=False,
                        data=None,
                        message=f"Error loading CSV: {str(e)}",
                        agent_name=self.name,
                    )

        # Build prompt for Gemini
        prompt = self._build_prompt(query, context)

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Extract and execute code blocks
            code_blocks = self._extract_code_blocks(response_text)

            if not code_blocks:
                return AgentResult(
                    success=True,
                    data={"analysis": response_text},
                    message="Analysis completed",
                    agent_name=self.name,
                    next_agent="AnswerSynthesiser",
                )

            # Execute code
            execution_results = []
            for code in code_blocks:
                result = self._execute_code(code)
                execution_results.append(result)

            return AgentResult(
                success=True,
                data={
                    "analysis": response_text,
                    "code_executed": code_blocks,
                    "results": execution_results,
                },
                message="Code execution completed",
                agent_name=self.name,
                next_agent="AnswerSynthesiser",
            )

        except Exception as e:
            return AgentResult(
                success=False,
                data={"error": str(e)},
                message=f"Error: {str(e)}",
                agent_name=self.name,
            )

    def _build_prompt(self, query: str, context: Dict[str, Any]) -> str:
        prompt = f"""You are a Python data analysis expert. Analyze the user's query and provide Python code.

User Query: {query}

"""
        if self.dataframes:
            prompt += "Available DataFrames (ALREADY LOADED):\n"
            for name, df in self.dataframes.items():
                prompt += f"\nVariable: {name}\n"
                prompt += f"  Shape: {df.shape}\n"
                prompt += f"  Columns: {df.columns.tolist()}\n"
                prompt += f"  Sample:\n{df.head().to_string()}\n"

        # Check if this is an edit operation
        is_edit_mode = context.get("edit_mode", False)
        output_file = context.get("output_file")
        
        if is_edit_mode and output_file:
            prompt += f"""
IMPORTANT - CSV EDIT MODE:
- You are editing a CSV file
- The DataFrame is already loaded (DO NOT use pd.read_csv())
- After making your edits, you MUST save the DataFrame using: df.to_csv('{output_file}', index=False)
- Make sure to save the file at the end of your code
- The output file path is: '{output_file}'
"""
        
        prompt += """
Instructions:
1. DataFrames are ALREADY LOADED - DO NOT use pd.read_csv() unless loading a new file
2. Write Python code in blocks
3. Use pandas (pd) and numpy (np)
4. Print results so they can be displayed
5. Don't create visualizations
"""
        
        if is_edit_mode:
            prompt += "6. CRITICAL: Save the edited DataFrame to the output file path using to_csv()\n"
        
        prompt += """
Provide your analysis and code:
"""
        return prompt

    def _extract_code_blocks(self, text: str) -> List[str]:
        code_blocks = []
        lines = text.split("\n")
        in_code_block = False
        current_block = []

        for line in lines:
            if line.strip().startswith(""):
                in_code_block = True
                current_block = []
            elif line.strip() == "```" and in_code_block:
                in_code_block = False
                if current_block:
                    code_blocks.append("\n".join(current_block))
            elif in_code_block:
                current_block.append(line)

        return code_blocks

    def _execute_code(self, code: str) -> Dict[str, Any]:
        # Create safe environment with dataframes
        safe_dataframes = {}
        for name, df in self.dataframes.items():
            safe_name = name.replace(".csv", "").replace("-", "_").replace(" ", "_")
            safe_dataframes[safe_name] = df
            safe_dataframes[name] = df

        exec_globals = {
            "pd": pd,
            "np": np,
            **safe_dataframes,
        }

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = {
            "code": code,
            "success": False,
            "output": "",
            "error": None,
        }

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_globals)

            result["success"] = True
            result["output"] = stdout_capture.getvalue()
            
            # Check if any dataframes were modified (for edit operations)
            # This helps track if the edit was successful
            modified_dataframes = {}
            for name, df in self.dataframes.items():
                # Check if dataframe still exists in exec_globals (might have been modified)
                if name in exec_globals:
                    modified_dataframes[name] = exec_globals[name]

        except Exception as e:
            result["error"] = str(e)
            result["output"] = stderr_capture.getvalue()

        return result
