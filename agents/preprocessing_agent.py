from typing import Dict
import re
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class PreprocessingAgent:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model="mixtral-8x7b-32768",
            temperature=0.1
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in preprocessing and normalizing text content.
            Your task is to:
            1. Clean and normalize the text
            2. Identify key phrases and terms
            3. Extract relevant context
            4. Flag potential triggers or sensitive content
            
            Maintain the essential meaning while removing noise."""),
            ("human", "{text}")
        ])
        
    def preprocess(self, text: str) -> Dict[str, str]:
        """
        Preprocess and normalize the input text
        
        Args:
            text: The text to preprocess
            
        Returns:
            Dict containing preprocessed text and extracted information
        """
        # Basic cleaning
        cleaned_text = text.strip()
        cleaned_text = re.sub(r'http\S+|www.\S+', '', cleaned_text)  # Remove URLs
        cleaned_text = re.sub(r'@\w+', '@user', cleaned_text)  # Anonymize usernames
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
        
        # Get additional analysis from LLM
        messages = self.prompt.format_messages(text=cleaned_text)
        response = self.llm.invoke(messages)
        
        # Parse the response into structured format
        result = {
            "preprocessed_text": cleaned_text,
            "key_phrases": [],
            "context": "",
            "trigger_warnings": []
        }
        
        # Extract relevant sections from the response
        current_section = ""
        for line in response.content.split("\n"):
            if "key phrase" in line.lower():
                current_section = "key_phrases"
            elif "context:" in line.lower():
                current_section = "context"
            elif "trigger" in line.lower() or "warning" in line.lower():
                current_section = "trigger_warnings"
            elif current_section and line.strip():
                if current_section == "key_phrases":
                    if line.strip().startswith("-"):
                        result["key_phrases"].append(line.strip()[1:].strip())
                elif current_section == "context":
                    result["context"] += line.strip() + " "
                elif current_section == "trigger_warnings":
                    if line.strip().startswith("-"):
                        result["trigger_warnings"].append(line.strip()[1:].strip())
                
        return result 