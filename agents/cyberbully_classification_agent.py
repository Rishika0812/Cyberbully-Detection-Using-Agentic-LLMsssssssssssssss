from typing import Dict, List
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class CyberbullyClassificationAgent:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model="mixtral-8x7b-32768",
            temperature=0.1
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in classifying cyberbullying content.
            Analyze the given text and classify it into the following categories:
            - Direct Harassment
            - Hate Speech
            - Identity-based Bullying
            - Threat
            - Not Cyberbullying
            
            For each category, provide a confidence score (0-1) and justification.
            Be objective and thorough in your analysis."""),
            ("human", "{text}")
        ])
        
    def classify(self, text: str) -> Dict[str, List[Dict]]:
        """
        Classify the type of cyberbullying in the given text
        
        Args:
            text: The text to classify
            
        Returns:
            Dict containing classification results with confidence scores
        """
        messages = self.prompt.format_messages(text=text)
        response = self.llm.invoke(messages)
        
        # Parse the response into structured format
        classifications = {
            "categories": [],
            "primary_category": "",
            "confidence": 0.0
        }
        
        # Process each line of the response
        for line in response.content.split("\n"):
            if any(category in line.lower() for category in 
                ["direct harassment", "hate speech", "identity-based", "threat", "not cyberbullying"]):
                
                # Extract category and confidence score
                parts = line.split(":")
                if len(parts) >= 2:
                    category = parts[0].strip()
                    details = parts[1].strip()
                    
                    # Extract confidence score if present
                    confidence = 0.0
                    if "(" in details and ")" in details:
                        try:
                            confidence = float(details.split("(")[1].split(")")[0])
                        except:
                            confidence = 0.0
                            
                    # Add to categories list
                    classifications["categories"].append({
                        "category": category,
                        "confidence": confidence,
                        "justification": details
                    })
                    
                    # Update primary category if confidence is higher
                    if confidence > classifications["confidence"]:
                        classifications["confidence"] = confidence
                        classifications["primary_category"] = category
        
        return classifications 