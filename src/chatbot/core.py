import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.chains import RetrievalQA
# Load environment variables
load_dotenv()

class ProductSearchTool(BaseTool):
    name: str = "product_search"
    description: str = "Tool to search for products in the database. Use it when the user asks about specific products or wants recommendations."
    df_products: Any = Field(default=None, description="DataFrame with product information")
    model: Any = Field(default=None, description="Embedding model")
    index: Any = Field(default=None, description="Search index")
    search_index: Any = Field(default=None, description="Search index function")
    
    def _run(self, query: str) -> str:
        try:
            indices, scores = self.search_index(self.index, self.model, query, k=5)
            productos = self.df_products.iloc[indices][['title', 'price', 'average_rating']]
            productos['relevancia'] = scores
            return f"Products found:\n{productos.to_string()}"
        except Exception as e:
            return f"Error searching products: {str(e)}"

class EcommerceAssistant:
    def __init__(
        self,
        df_products = None,
        model = None,
        index = None,
        search_index = None,
        retriever = None
    ):
        # Initialize message history
        self.messages: List[BaseMessage] = []
        
        # Initialize tools
        self.tools = [
            ProductSearchTool(
                df_products=df_products,
                model=model,
                index=index,
                search_index=search_index
            )
        ] if all(x is not None for x in [df_products, model, index, search_index]) else []
        
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        
        # Prompt para el sistema
        self.system_prompt = """You are an expert in e-commerce customer service.
        Your goal is to help users with their e-commerce related questions.
        
        Instructions:
        1. Carefully analyze the user's query
        2. If the query is about products, use the product_search tool to find relevant products
        3. Present product information in a clear and organized manner, including:
           - Product names
           - Prices
           - Ratings
           - Relevance scores
        4. If no products are found or the results are not satisfactory:
           - Suggest alternative search terms
           - Ask for more specific information
        5. For non-product queries, provide helpful and accurate information
        6. Always maintain a professional and friendly tone
        7. Remember previous interactions to provide context-aware responses
        
        Guidelines:
        - Be clear and concise in your responses
        - If you don't know something, be honest about it
        - Suggest relevant alternatives when appropriate
        - Keep the conversation focused on e-commerce topics
        - When showing products, always mention price and rating
        - If the user's query is unclear, ask for more details
        
        IMPORTANTE: Cuando uses la herramienta product_search, DEBES incluir en tu respuesta la frase "He usado la herramienta de búsqueda para encontrar estos productos:" antes de mostrar los resultados.
        
        Available tools:
        {tools}
        
        Tool names: {tool_names}"""
        
        # Prompt for the chat
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            *self.messages,
            ("human", "{text}")
        ])
        
        # # Initialize chain
        # self.chain = (
        #     self.chat_prompt 
        #     | self.llm 
        #     | StrOutputParser()
        # )

        self.chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)

    
    def process_query(self, query: str) -> str:
        try:
            # Add user message to history
            self.messages.append(HumanMessage(content=query))
            
            # Use the chain to process the query
            result = self.chain.run(query)
            
            # Add assistant response to history
            self.messages.append(AIMessage(content=result))
            
            return result
                
        except Exception as e:
            return f"Sorry, there was an error processing your query: {str(e)}"

def chatbot_response(
    query: str,
    assistant: EcommerceAssistant,
) -> str:
    """
    Wrapper function to maintain compatibility with existing code
    """
    return assistant.process_query(query)
    