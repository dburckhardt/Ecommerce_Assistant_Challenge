from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader

class ProductRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize RAG with model
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0
        )
        
    def create_vectorstore(self, df_products):
        """
        Create vectorstore from dataframe
        """
        df_products['combined_content'] = df_products.apply(
            lambda row: (
                f"Category: {row['main_category']} | "
                f"Product: {row['title']} | "
                f"Rating: {row['average_rating']} stars | "
                f"Description: {row['description']}"
            ),
            axis=1
        )
        loader = DataFrameLoader(df_products, page_content_column='combined_content')
        documents = loader.load()
        
        # Create vectorstore
        self.vectorstore = FAISS.from_documents(
            documents,
            self.embeddings
        )
        
        # Configure basic retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
    
    def search(self, query, k=5):
        """
        Search for relevant documents
        """
        return self.retriever.get_relevant_documents(query)