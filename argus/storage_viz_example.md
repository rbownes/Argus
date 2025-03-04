from llm_storage import setup_storage, import_from_chromadb

# Initialize storage
storage_manager = setup_storage(base_dir="./llm_data")

# Import existing data from ChromaDB
import_from_chromadb(storage_manager, collection, max_batches=10)

# Store metrics results
metrics_results = [...]  # Your metric results
storage_manager.store_metrics(metrics_results)

# Run the setup script
python setup_grafana.py

# Start the containers
cd ./monitoring && docker-compose up -d

# Access Grafana at http://localhost:3000
# Username: admin, Password: admin

# Install Streamlit
pip install streamlit nltk plotly

# Run the dashboard
streamlit run streamlit_dashboard.py