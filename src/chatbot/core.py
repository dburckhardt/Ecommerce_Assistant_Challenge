import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from api_client import OrderAPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()

class EcommerceAssistant:
    def __init__(
        self,
        df_products = None,
        search_index = None,
    ):
        # Initialize message history
        self.messages: List[BaseMessage] = []
        
        # Initialize OrderAPI
        self.order_api = OrderAPI()
        
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0
        )
        
        # Initialize tools
        self.tools = []
        
        if all(x is not None for x in [df_products, search_index]):
            @tool
            def search_products(query: str) -> str:
                """Search for products in the database.
                Input should be a search query about products.
                Returns a list of relevant products with their titles, prices, and ratings.
                Use this when the user asks about specific products or wants recommendations."""
                try:
                    indices, scores = search_index(query, k=5)
                    productos = df_products.iloc[indices][['title', 'price', 'average_rating']]
                    productos['relevancia'] = scores
                    return f"Products found:\n{productos.to_string()}"
                except Exception as e:
                    return f"Error searching products: {str(e)}"
            
            self.tools.append(search_products)
        
        @tool
        def get_order(customer_id: str) -> str:
            """Get order information for a specific customer ID.
            Input should be a customer ID number.
            Returns the order details if found, or an error message if not found.
            Use this when the user asks about their order status or order details."""
            print(f"Getting order for customer ID: 37077")
            try:
                result = self.order_api.get_order_by_id(37077)
                if result["error"]:
                    return f"Error retrieving order: {result['error']}"
                if not result["orders"]:
                    return f"No orders found for customer ID: {customer_id}"
                return f"Order details:\n{result['orders']}"
            except ValueError:
                return "Please provide a valid customer ID number"
            except Exception as e:
                return f"Error processing order request: {str(e)}"
        
        self.tools.append(get_order)
        
        # System prompt for the agent
        self.system_prompt = """You are an expert e-commerce customer service assistant.
        Your goal is to help users with their e-commerce related questions.
        
        Instructions:
        1. Carefully analyze the user's query
        2. If the query is about products, use the search_products tool to find relevant products
        3. If the query is about order status or details, use the get_order tool with the customer ID
        4. Present product information in a clear and organized manner, including:
           - Product names
           - Prices
           - Ratings
           - Relevance scores
        5. For order information, present:
           - Order details
           - Status
           - Any relevant error messages if the order is not found
        6. If no products are found or the results are not satisfactory:
           - Suggest alternative search terms
           - Ask for more specific information
        7. For non-product queries, provide helpful and accurate information
        8. Always maintain a professional and friendly tone
        9. Remember previous interactions to provide context-aware responses
        
        Guidelines:
        - Be clear and concise in your responses
        - If you don't know something, be honest about it
        - Suggest relevant alternatives when appropriate
        - Keep the conversation focused on e-commerce topics
        - When showing products, always mention price and rating
        - When showing order information, present it in a clear and organized way
        - If the user's query is unclear, ask for more details
        
        IMPORTANT: 
        - When using the search_products tool, you MUST include the phrase "I have used the search tool to find these products:" before showing the results.
        - When using the get_order tool, you MUST include the phrase "I have retrieved the order information:" before showing the results."""
        
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            system_message=self.system_prompt
        )
    
    def process_query(self, query: str) -> str:
        try:
            # Add user message to history
            self.messages.append(HumanMessage(content=query))
            
            # Use the agent to process the query and get only the final answer
            result = self.agent.invoke({"input": query})["output"]
            
            # Add assistant response to history
            self.messages.append(AIMessage(content=result))
            
            return result
                
        except Exception as e:
            error_msg = f"sorry, there was an error processing your query: {str(e)}"
            self.messages.append(AIMessage(content=error_msg))
            return error_msg

def chatbot_response(
    query: str,
    assistant: EcommerceAssistant,
) -> str:
    """
    Wrapper function to maintain compatibility with existing code
    """
    return assistant.process_query(query)
    