# E-commerce Assistant

An intelligent chatbot that uses RAG (Retrieval Augmented Generation) to answer queries about products and orders in an e-commerce system.

## Main Features

- Semantic search for products and orders
- Integration with mock API for product and order data
- Specific tools for order queries by ID and priority
- LLM-based product ranking system
- Interactive web interface using Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dburckhardt/Ecommerce_Assistant_Challenge.git
cd Ecommerce_Assistant_Challenge
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate ecommerce-chatbot
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with the necessary credentials
```

4. Start the mock API:
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

3. Open your browser at the URL shown in the terminal (usually http://localhost:8501)

The Streamlit interface provides:
- A modern chat interface
- Real-time conversation history
- Example questions in the sidebar
- Clear conversation button
- Automatic assistant initialization

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

The system implements a hybrid approach for product ranking:

1. **Semantic Search**: Finds products relevant to the query
2. **LLM Reordering**: Reorders results based on specific criteria
3. **Contextual Ratings**: Ratings are available to the LLM but not used for direct search

### ID Handling Security

The current implementation has security limitations:

- Does not verify user identity
- Allows access to any order without validation
- Does not implement authentication

In production, it should:
- Implement user authentication
- Validate access permissions
- Restrict access to sensitive information
- Get IDs from session context

### Known Limitations

1. **Priority Search**:
   - Works for 'low', 'medium', and 'critical'
   - Does not find results for 'high'
   - Can be verified in `notebooks/api_client.ipynb` where API methods are consumed for testing.

2. **Cell Phone Orders Use Case**:
   - Current dataset does not contain multiple cell phone orders for any customer
   - The query "What is the status of my cell-phone order?" may return unrelated orders
   - Pending improvement: Implement validation in the system prompt to verify the relevance of found orders with respect to the user's query

3. **Unimplemented Features**:
   - Additional API endpoints
   - Challenge bonus features

### Tested Models

- Llama 3.2 8b (Local)
- Gemini (Free API with models like Gemini 1.5 flash, 2.0 flash)
- Mistral (Discarded due to token limit limitations)

## Pending Improvements

1. **Performance**:
   - Implement response streaming
   - Optimize semantic search
   - Add progress indicators

2. **Functionality**:
   - Improve ranking system
   - Add authentication
   - Add tools for other API endpoints
