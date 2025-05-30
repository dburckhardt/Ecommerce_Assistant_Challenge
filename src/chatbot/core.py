import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from api_client import OrderAPI
import logging
import pandas as pd
from rag import ProductRAG
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EcommerceAssistant:
    def __init__(
        self,
    ):
        # Load products data
        csv_path = Path(__file__).parent.parent.parent / 'data' / 'Product_Information_Dataset.csv'
        print(csv_path)
        logger.info(f"Intentando cargar el archivo CSV desde: {csv_path}")
        df_products = pd.read_csv(csv_path)
        rag = ProductRAG()
        rag.create_vectorstore(df_products)

        def search_index(query, k=5):
            results = rag.search(query, k=k)
            # Use the document index instead of metadata
            indices = [i for i in range(len(results))]
            scores = [doc.metadata.get('score', 0.0) for doc in results]
            return indices, scores

        # Initialize message history
        self.messages: List[BaseMessage] = []
        
        # Initialize OrderAPI
        self.order_api = OrderAPI()
        
        # Initialize Mistral model
        # self.llm = ChatMistralAI(
        #     model="mistral-large-latest",
        #     mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        #     temperature=0.0
        # )

                # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0
        )
        
        # Initialize tools
        self.tools = []
        @tool
        def search_products(query: str) -> str:
            """Search for products in the database.
            Input should be a search query about products.
            Returns a list of relevant products with their titles, prices, and ratings.
            Use this when the user asks about specific products or wants recommendations."""
            try:
                indices, scores = search_index(query, k=5)
                productos = df_products.iloc[indices][['main_category', 'title', 'price', 'average_rating']]
                productos['relevancia'] = scores
                tool_response = f"Products found:\n{productos.to_string()}"
                return tool_response
            except Exception as e:
                error_msg = f"Error searching products: {str(e)}"
                return error_msg
        
        self.tools.append(search_products)
        
        @tool
        def get_order(customer_id: str) -> str:
            """Get order information for a specific customer ID.
            Input should be a customer ID number.
            Returns the order details if found, or an error message if not found.
            Use this when the user asks about their order status or order details."""
            try:
                result = self.order_api.get_order_by_id(customer_id)
                if result["error"]:
                    tool_response = f"Error retrieving order: {result['error']}"
                    return tool_response
                if not result["orders"]:
                    tool_response = f"No orders found for customer ID: {customer_id}"
                    return tool_response
                tool_response = f"Order details:\n{result['orders']}"
                return tool_response
            except ValueError:
                tool_response = "Please provide a valid customer ID number"
                return tool_response
            except Exception as e:
                tool_response = f"Error processing order request: {str(e)}"
                return tool_response
        
        self.tools.append(get_order)
        
        @tool
        def get_orders_by_priority(priority: str) -> str:
            """Get orders by priority level.
            Input should be one of: 'high', 'medium', 'low', 'critical'.
            Returns a list of orders with the specified priority level.
            Use this when the user asks about orders with specific priority levels."""
            try:
                # Normalizar la prioridad a minúsculas
                priority = priority.lower()
                valid_priorities = ['high', 'medium', 'low', 'critical']
                
                if priority not in valid_priorities:
                    return f"Invalid priority level. Please use one of: {', '.join(valid_priorities)}"
                
                result = self.order_api.get_order_by_priority(priority)
                if result["error"]:
                    tool_response = f"Error retrieving orders: {result['error']}"
                    return tool_response
                if not result["orders"]:
                    tool_response = f"No orders found with priority level: {priority}"
                    return tool_response
                
                # Formatear la respuesta para mejor legibilidad
                orders = result["orders"]
                tool_response = f"Orders with {priority} priority:\n\n" + f"\n\n {orders}"
                return tool_response
            except Exception as e:
                tool_response = f"Error processing priority request: {str(e)}"
                return tool_response
        
        self.tools.append(get_orders_by_priority)
        
        # System prompt for the agent
        self.system_prompt = """You are an expert e-commerce customer service assistant.
        Your goal is to help users with their e-commerce related questions.
        
        Instructions:
        1. Carefully analyze the user's query and the conversation history
        2. If the query is about products, use the search_products tool to find relevant products
        3. If the query is about order status or details:
           - For specific customer orders, use the get_order tool with the customer ID and give all the information about the order
           - For orders by priority level, use the get_orders_by_priority tool
        4. If no products are found or the results are not satisfactory:
           - Suggest alternative search terms
           - Ask for more specific information
        5. For non-product queries, provide helpful and accurate information
        6. Always maintain a professional and friendly tone
        7. Use the conversation history to:
           - Refer to previous questions or products mentioned
           - Maintain context about user preferences
           - Provide more relevant follow-up suggestions
           - Avoid repeating information already provided
        
        Guidelines:
        - Be clear and concise in your responses
        - If you don't know something, be honest about it
        - Suggest relevant alternatives when appropriate
        - Keep the conversation focused on e-commerce topics
        - When showing order information, present it in a clear and organized way
        - If the user's query is unclear, ask for more details
        - For priority-based queries, ensure to use the correct priority levels: high, medium, low, critical"""
        
        self.messages.append(SystemMessage(content=self.system_prompt))
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            system_message=self.system_prompt,
            return_intermediate_steps=False,
        )
    
    def process_query(self, query: str) -> str:
        try:
            # Add user message to history
            self.messages.append(HumanMessage(content=query))
            
            # Use the agent to process the query and get only the final answer
            #result = self.agent.invoke({"input": self.messages})
            result = self.agent.invoke({"input": self.messages})
            #print(result['intermediate_steps'])
            # Add assistant response to history
            self.messages.append(AIMessage(content=result["output"]))
            
            return result["output"]
                
        except Exception as e:
            error_msg = f"sorry, there was an error processing your query: {str(e)}"
            self.messages.append(AIMessage(content=error_msg))
            return error_msg
        
    def reset_messages(self):
        self.messages = []
        self.messages.append(SystemMessage(content=self.system_prompt))

def chatbot_response(
    query: str,
    assistant: EcommerceAssistant,
) -> str:
    """
    Wrapper function to maintain compatibility with existing code
    """
    return assistant.process_query(query)

def clear_conversation(assistant: EcommerceAssistant):
    assistant.reset_messages()
    