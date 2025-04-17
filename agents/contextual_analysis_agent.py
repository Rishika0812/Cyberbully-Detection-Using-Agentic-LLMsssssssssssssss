from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class ContextualAnalysisAgent:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model="mixtral-8x7b-32768",
            temperature=0.3
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in contextual analysis of cyberbullying content.
            Your task is to analyze the given text and provide:
            1. The broader social context of the bullying
            2. Potential underlying motivations
            3. Cultural or social factors that might be relevant
            4. Historical context if applicable
            
            Be thorough but sensitive in your analysis. Focus on understanding rather than judgment."""),
            ("human", "{text}")
        ])

    def analyze(self, text: str) -> Dict[str, str]:
        """
        Analyze the context of potentially bullying content
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing analysis results
        """
        messages = self.prompt.format_messages(text=text)
        response = self.llm.invoke(messages)
        
        # Parse the response into structured format
        analysis = {
            "social_context": "",
            "motivations": "",
            "cultural_factors": "",
            "historical_context": ""
        }
        
        # Extract relevant sections from the response
        current_section = ""
        for line in response.content.split("\n"):
            if "social context:" in line.lower():
                current_section = "social_context"
            elif "motivation" in line.lower():
                current_section = "motivations"
            elif "cultural" in line.lower():
                current_section = "cultural_factors"
            elif "historical" in line.lower():
                current_section = "historical_context"
            elif current_section and line.strip():
                analysis[current_section] += line.strip() + " "
                
        return analysis 