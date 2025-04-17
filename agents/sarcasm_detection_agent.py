from typing import Dict
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class SarcasmDetectionAgent:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model="mixtral-8x7b-32768",
            temperature=0.2
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in detecting sarcasm and subtle forms of cyberbullying.
            Your task is to:
            1. Identify sarcastic or passive-aggressive content
            2. Analyze the true intent behind the words
            3. Detect subtle forms of mockery or ridicule
            4. Evaluate the potential psychological impact
            
            Be thorough in identifying hidden meanings and subtle forms of bullying."""),
            ("human", "{text}")
        ])
        
    def detect_sarcasm(self, text: str) -> Dict[str, any]:
        """
        Detect and analyze sarcastic content in text
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing sarcasm analysis results
        """
        messages = self.prompt.format_messages(text=text)
        response = self.llm.invoke(messages)
        
        # Parse the response into structured format
        analysis = {
            "is_sarcastic": False,
            "confidence": 0.0,
            "sarcasm_type": "",
            "true_intent": "",
            "subtle_indicators": [],
            "psychological_impact": ""
        }
        
        # Extract relevant sections from the response
        current_section = ""
        for line in response.content.split("\n"):
            if "sarcasm detection:" in line.lower():
                # Try to extract confidence score
                if "(" in line and ")" in line:
                    try:
                        confidence = float(line.split("(")[1].split(")")[0])
                        analysis["confidence"] = confidence
                        analysis["is_sarcastic"] = confidence > 0.5
                    except:
                        pass
            elif "type:" in line.lower():
                analysis["sarcasm_type"] = line.split(":")[1].strip()
            elif "true intent:" in line.lower():
                current_section = "true_intent"
            elif "indicator" in line.lower():
                current_section = "subtle_indicators"
            elif "psychological impact:" in line.lower():
                current_section = "psychological_impact"
            elif current_section and line.strip():
                if current_section == "subtle_indicators":
                    if line.strip().startswith("-"):
                        analysis["subtle_indicators"].append(line.strip()[1:].strip())
                elif current_section == "true_intent":
                    analysis["true_intent"] += line.strip() + " "
                elif current_section == "psychological_impact":
                    analysis["psychological_impact"] += line.strip() + " "
                
        return analysis 