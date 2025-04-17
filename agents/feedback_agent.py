from typing import Dict, List
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class FeedbackAgent:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model="mixtral-8x7b-32768",
            temperature=0.3
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in providing constructive feedback and recommendations 
            for addressing cyberbullying content. Your task is to:
            1. Suggest appropriate responses to the bullying
            2. Provide support resources for victims
            3. Recommend preventive measures
            4. Suggest educational resources
            
            Be supportive, practical, and action-oriented in your recommendations."""),
            ("human", """Text: {text}
            Classification: {classification}
            Analysis: {analysis}
            
            Please provide feedback and recommendations.""")
        ])
        
    def get_feedback(self, text: str, classification: Dict, analysis: Dict) -> Dict[str, List[str]]:
        """
        Generate feedback and recommendations for cyberbullying content
        
        Args:
            text: The original text
            classification: Classification results
            analysis: Contextual analysis results
            
        Returns:
            Dict containing structured feedback and recommendations
        """
        messages = self.prompt.format_messages(
            text=text,
            classification=classification,
            analysis=analysis
        )
        response = self.llm.invoke(messages)
        
        # Parse the response into structured format
        feedback = {
            "immediate_responses": [],
            "support_resources": [],
            "preventive_measures": [],
            "educational_resources": []
        }
        
        # Extract relevant sections from the response
        current_section = ""
        for line in response.content.split("\n"):
            if "immediate response" in line.lower():
                current_section = "immediate_responses"
            elif "support" in line.lower():
                current_section = "support_resources"
            elif "prevent" in line.lower():
                current_section = "preventive_measures"
            elif "education" in line.lower():
                current_section = "educational_resources"
            elif current_section and line.strip():
                # Add bullet points as separate items
                if line.strip().startswith("-"):
                    feedback[current_section].append(line.strip()[1:].strip())
                else:
                    feedback[current_section].append(line.strip())
                
        return feedback 