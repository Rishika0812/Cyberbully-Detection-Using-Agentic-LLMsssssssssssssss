from typing import Dict
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class ExplainabilityAgent:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model="mixtral-8x7b-32768",
            temperature=0.2
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in explaining cyberbullying content analysis in clear, 
            understandable terms. Your task is to:
            1. Explain why the content is classified as cyberbullying
            2. Break down the key indicators and patterns
            3. Provide clear examples of similar patterns
            4. Explain potential impacts
            
            Make your explanations accessible to both technical and non-technical audiences."""),
            ("human", """Text: {text}
            Classification: {classification}
            Analysis: {analysis}
            
            Please provide a clear explanation.""")
        ])
        
    def explain(self, text: str, classification: Dict, analysis: Dict) -> Dict[str, str]:
        """
        Generate clear explanations for cyberbullying content analysis
        
        Args:
            text: The original text
            classification: Classification results
            analysis: Contextual analysis results
            
        Returns:
            Dict containing structured explanations
        """
        messages = self.prompt.format_messages(
            text=text,
            classification=classification,
            analysis=analysis
        )
        response = self.llm.invoke(messages)
        
        # Parse the response into structured format
        explanation = {
            "classification_explanation": "",
            "key_indicators": "",
            "similar_patterns": "",
            "potential_impacts": ""
        }
        
        # Extract relevant sections from the response
        current_section = ""
        for line in response.content.split("\n"):
            if "classification:" in line.lower():
                current_section = "classification_explanation"
            elif "indicator" in line.lower():
                current_section = "key_indicators"
            elif "pattern" in line.lower():
                current_section = "similar_patterns"
            elif "impact" in line.lower():
                current_section = "potential_impacts"
            elif current_section and line.strip():
                explanation[current_section] += line.strip() + " "
                
        return explanation 