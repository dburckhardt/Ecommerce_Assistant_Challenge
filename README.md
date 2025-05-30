# E-commerce Assistant

An intelligent chatbot that uses RAG (Retrieval Augmented Generation) to answer queries about products and orders in an e-commerce system. The assistant combines the power of large language models with semantic search capabilities to provide accurate and contextually relevant responses to customer inquiries.

## Main Features

The system offers a comprehensive suite of features designed to enhance the e-commerce customer service experience. At its core, it provides semantic search capabilities for both products and orders, seamlessly integrated with a mock API for real-time data access. The system includes specialized tools for order queries, supporting both ID-based lookups and priority-based filtering. A sophisticated LLM-based product ranking system ensures that customers receive the most relevant product recommendations.

## Installation

To get started with the E-commerce Assistant, follow these steps:

First, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/dburckhardt/Ecommerce_Assistant_Challenge.git
cd Ecommerce_Assistant_Challenge
```

Next, set up the development environment using conda:
```bash
conda env create -f environment.yml
conda activate ecommerce-chatbot
```

Configure your environment variables by copying the example file and editing it with your credentials:
```bash
cp .env.example .env
# Edit .env with the necessary credentials
```

Finally, start the mock API server:
```bash
cd mock_api
pip install -r requirements.txt
uvicorn mock_api:app --reload --port 8000
```
The mock API will be available at http://localhost:8000

## Usage

### Web Interface (Streamlit)

1. Make sure the mock API is running (see Installation step 4)
2. Start the Streamlit app:
```bash
streamlit run src/streamlit_app.py
```

The application will be accessible through your browser at the URL shown in the terminal (typically http://localhost:8501).

### Notebook Interface

1. Activate the environment:
```bash
conda activate ecommerce-chatbot
jupyter notebook
```

2. Open `notebooks/chatbot.ipynb` to interact with the assistant.

### Test Notebooks

- `notebooks/test_orders.ipynb`: Order query tests
- `notebooks/test_products.ipynb`: Product query tests

## Project Structure

```
Ecommerce_Assistant_Challenge/
├── data/                    # Product and order data
├── notebooks/              # Jupyter notebooks for development and testing
├── src/                    # Project source code
│   ├── chatbot/           # Main chatbot module
│   ├── api_client.py      # Client for API interaction
│   ├── rag.py             # RAG implementation
│   └── streamlit_app.py   # Streamlit web interface
├── mock_api/              # Mock API for development
├── environment.yml        # Conda environment configuration
└── .env                   # Environment variables (not versioned)
```

## Important Notes

### Rating and Ranking Management

The system employs a sophisticated hybrid approach for product ranking that combines multiple strategies. First, semantic search identifies products relevant to the query. Then, an LLM-based system reorders these results based on specific criteria. While ratings are available to the LLM for context, they are not used directly in the search process, allowing for more nuanced ranking decisions.

### ID Handling Security

The current implementation has several security considerations that should be addressed before production deployment. The system currently operates without user identity verification and allows access to any order without validation. In a production environment, it should implement proper user authentication, validate access permissions, restrict sensitive information access, and obtain IDs from session context.

### Known Limitations

The system has several known limitations that users should be aware of:

**Priority Search**: The system currently supports 'low', 'medium', and 'critical' priority levels, but does not find results for 'high' priority. This can be verified in the `notebooks/api_client.ipynb` where API methods are tested.

**Cell Phone Orders**: The current dataset lacks multiple cell phone orders for any customer. As a result, queries about cell phone order status may return unrelated orders. A future improvement will implement validation in the system prompt to verify order relevance. The query "What is the status of my cell-phone order?" may return unrelated orders because don't exist the answer show at the challenge in the dataset provided.

**Feature Implementation**: Several features remain unimplemented, including additional API endpoints and challenge bonus features. These will be addressed in future updates.

## Model Analysis and Trade-offs

### Evaluated Models

The system has been tested with several language models, each offering different advantages and trade-offs:

1. **Llama 3.2 8B (Local)**
   - **Advantages**:
     - Minimal inference cost (infrastructure costs only)
     - Full implementation control
     - No API limits
   - **Disadvantages**:
     - Inconsistent response quality
     - Overly concise responses
     - Requires significant local infrastructure

2. **Mistral**
   - **Advantages**:
     - Good performance on general tasks
     - Competitive cost
   - **Disadvantages**:
     - Critical token limitations in complex queries
     - Stability issues with extensive order queries
     - Not suitable for specific use case
   - **Cost**: ~$0.20-0.30 per million tokens

3. **Gemini (API)**
   - **Evaluated Versions**:
     - Gemini 1.5 Flash
     - Gemini 2.0 Flash
     - Gemini 2.5 Flash
   - **Advantages**:
     - Better response quality
     - Consistent performance
     - Immediate scalability
     - No token issues
   - **Disadvantages**:
     - API costs
     - Connectivity dependency
   - **Costs**:
     - Gemini 1.5: ~$0.075 per million tokens input
     - Gemini 2.0: ~$0.10 per million tokens input
     - Gemini 2.5: ~$0.50 per million tokens input

### Trade-off Analysis

The model selection process involved careful consideration of several key factors:

**Quality vs. Cost**: Local models like Llama require higher initial GPU costs but lower operational costs, though with inferior response quality. Cloud models like Gemini eliminate initial costs but have higher operational costs, delivering superior response quality.

**Inference Time**: Local models typically show 2-3 seconds latency and are hardware-dependent, while cloud models offer 0.5-1.5 seconds latency with consistent performance regardless of hardware.

**Scalability**: Local models are limited by hardware constraints and require resource management, whereas cloud models provide automatic scalability without infrastructure management overhead.

### Final Decision

After thorough evaluation, Gemini was selected as the primary model for several compelling reasons: consistent response quality, immediate scalability, absence of token limitations, optimal response time, and a favorable Total Cost of Ownership (TCO) considering quality. The specific version (1.5, 2.0, or 2.5) is chosen based on query complexity, available budget, and latency requirements.

## Design Decisions and Technology Stack

### Core Technologies

The system leverages several key technologies to deliver its functionality:

**LangChain Framework** was chosen for its robust agent architecture and simplified LLM integration. It provides built-in tool management and structured chat capabilities, enabling rapid development of agent-based systems with standardized interfaces for different LLMs. The framework's built-in memory management and extensible tool system make it an ideal choice for this application.

**FAISS Vector Store** serves as the backbone of our semantic search implementation. This high-performance vector similarity search system enables fast and efficient product searches, scaling well to large product catalogs while maintaining memory efficiency and fast retrieval times.

**Streamlit Interface** was selected for its rapid UI development capabilities and built-in chat interface. The framework's session state management, chat message components, and automatic rerun on state changes make it perfect for our interactive application needs.

**LLM Integration** is implemented through a modular design that abstracts model switching through LangChain. The system uses a zero-shot learning approach with structured chat format, temperature control, and token management.

### Design Patterns

The system implements several key design patterns:

**Agent-Based Architecture** enables complex reasoning and tool usage while maintaining conversation context. This is implemented through a structured chat agent that makes tool-based decisions and provides context-aware responses.

**RAG Implementation** combines product data vectorization, semantic search, and context injection to deliver accurate product matching and contextual responses through efficient retrieval.

**API Integration** follows a clean separation of concerns approach, using a mock API for development with standardized response formats, comprehensive error handling, rate limiting, and response caching.

## Agent Architecture and Flow

### Core Components

The system's architecture centers around the `EcommerceAssistant` class, which orchestrates the entire operation. During initialization, it loads product data from CSV, initializes the ProductRAG system for semantic search, sets up message history, configures the Gemini model, initializes the OrderAPI client, and defines the necessary tools and system prompt.

The assistant provides three main tools:

**search_products** performs semantic search in the product database, returning the top 5 relevant products with their categories, titles, prices, ratings, and relevance scores.

**get_order** retrieves specific order information, validating customer IDs and returning detailed order information or appropriate error messages.

**get_orders_by_priority** filters orders by priority level, supporting 'high', 'medium', 'low', and 'critical' priorities, and returns a formatted list of matching orders.

The system prompt configuration defines the agent's behavior and capabilities, setting guidelines for query analysis, tool selection, response formatting, conversation management, and error handling.

### Processing Flow

The system follows a clear processing flow:

1. **Query Reception**: User queries are received and processed through a structured flow:
   ```mermaid
   graph TD
   A[User Query] --> B[Message History Update]
   B --> C[Agent Processing]
   C --> D[Tool Selection]
   D --> E[Response Generation]
   E --> F[History Update]
   ```

2. **Tool Selection Logic**
   - **Product Queries**:
     - Triggers `search_products`
     - Uses semantic search with RAG
     - Ranks results by relevance
   
   - **Order Status Queries**:
     - For specific orders: Uses `get_order`
     - For priority-based: Uses `get_orders_by_priority`
   
   - **Error Handling**:
     - Validates input parameters
     - Provides clear error messages
     - Maintains conversation context

## Pending Improvements

### Performance Optimizations

**Response Streaming** will be enhanced through chunked delivery using Server-Sent Events (SSE), client-side streaming buffers, and fallback mechanisms for non-streaming clients. Progress tracking for long-running queries and optimized token generation will improve the user experience.

**Semantic Search Enhancement** will include vector caching for frequently accessed embeddings, batch processing for concurrent queries, and optimized embedding model selection. A hybrid search approach will combine vector similarity search, keyword-based search, and metadata filtering, with query result caching using TTL.

**User Experience** improvements will add real-time progress indicators for query processing, search progress, and response generation. The system will implement estimated time remaining calculations and visual feedback for search depth, result relevance, and processing stages.

### Functional Enhancements

**Ranking System Improvements** will implement a multi-factor ranking algorithm considering semantic relevance, historical user preferences, product popularity, price competitiveness, and stock availability. The system will include A/B testing capabilities, feedback loops for optimization, and custom ranking rules for specific product categories.

**Authentication and Security** will be enhanced through OAuth 2.0 authentication flow, role-based access control, and comprehensive session management. The system will implement rate limiting, abuse prevention, and audit logging for security events.

### Infrastructure and Scalability

**Deployment Optimization** will include containerization using Docker, Kubernetes orchestration support, and a CI/CD pipeline with automated testing, deployment staging, and rollback capabilities. Infrastructure as code (IaC) will be implemented for better management.

**Monitoring and Observability** will be improved through a comprehensive logging system, metrics collection for response times and error rates, and an alerting system for performance degradation and resource constraints.

**Data Management** will be enhanced with data versioning, backup and recovery procedures, optimized database indexing, and data retention policies. Sensitive information will be properly anonymized.
