import os
from dotenv import load_dotenv
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


class ConfigLoader:
    def __init__(self):
        print(f"Loaded config.....")
        self.config = load_config()
    
    def __getitem__(self, key):
        return self.config[key]

class ModelLoader(BaseModel):
    model_provider: Literal["groq", "openai"] = "groq"
    config: Optional[ConfigLoader] = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        self.config = ConfigLoader()
    
    class Config:
        arbitrary_types_allowed = True
    
    def load_llm(self):
        """
        Load and return the LLM model.
        """
        print("LLM loading...")
        print(f"Loading model from provider: {self.model_provider}")
        load_dotenv(override=True)
        if self.model_provider == "groq":
            print("Loading LLM from Groq..............")
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is missing in .env file. Please enter a valid GROQ_API_KEY in .env file.")
            model_name = self.config["llm"]["groq"]["model_name"]
            llm = ChatGroq(model=model_name, api_key=groq_api_key)
        elif self.model_provider == "openai":
            print("Loading LLM from OpenAI..............")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is missing in .env file. Please enter a valid OPENAI_API_KEY in .env file.")
            model_name = self.config["llm"]["openai"]["model_name"]
            llm = ChatOpenAI(model_name=model_name, api_key=openai_api_key)
        
        return llm
    