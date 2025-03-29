Okay, I have modified the README to use `curl` commands for the API examples in the "User Journey" section.

```markdown
# Panopticon 🔭: LLM Evaluation & Monitoring System

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](...) <!-- Replace with your CI/CD badge -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Panopticon is a microservices-based system designed for the systematic evaluation, comparison, and monitoring of Large Language Model (LLM) performance.

## Overview & Intention

In the rapidly evolving landscape of LLMs, understanding how different models perform on *your specific tasks* is crucial. Panopticon provides a framework to:

1.  **Define Custom Evaluations:** Move beyond generic benchmarks. Create queries (prompts) relevant to your domain and define specific evaluation criteria (metrics/evaluation prompts) to measure what matters most to you.
2.  **Compare Models & Providers:** Systematically run the same queries and evaluations across different LLMs (e.g., GPT-4 vs. Claude 3 vs. Gemini Pro) or different versions of the same model.
3.  **Monitor Performance Over Time:** Track how model performance changes as models are updated or as you refine your prompts and evaluation strategies.
4.  **Visualize Results:** Gain insights through an integrated dashboard showing trends, comparisons, and detailed results.

Panopticon aims to provide an objective, configurable, and centralized platform for your LLM quality assurance and monitoring needs.

## ✨ Features

*   **Microservice Architecture:** Scalable and maintainable design using FastAPI and Docker.
*   **Configurable Evaluation:** Define your own queries (prompts) and evaluation metrics (judge prompts).
*   **Multi-Provider Support:** Integrates with various LLM providers via LiteLLM and custom adapters (OpenAI, Google Gemini, Anthropic included by default).
*   **Centralized Model Registry:** Manage available models and their configurations.
*   **Judge-Based Scoring:** Utilizes a powerful LLM (e.g., GPT-4) to score responses based on your criteria.
*   **Vector Storage & Search:** Stores queries and metrics with vector embeddings (using `sentence-transformers` and `pgvector`) for semantic similarity search.
*   **Data Persistence:** Uses PostgreSQL to store queries, metrics, model configurations, and detailed evaluation results.
*   **Visualization Dashboard:** A React-based frontend to explore evaluation trends, compare models, analyze themes, and view detailed results.
*   **API Gateway:** A central entry point (`main-app`) for interacting with the system.

## 🏗️ Architecture

Panopticon employs a microservice architecture:

*   **`main-app` (API Gateway):** The front door. Receives API requests and routes them to the appropriate backend service. Handles authentication.
*   **`item-storage-queries`:** Stores user-defined input queries/prompts, categorized by theme. Includes vector embeddings for search.
*   **`item-storage-metrics`:** Stores user-defined evaluation prompts/criteria, categorized by type. Includes vector embeddings for search.
*   **`judge-service`:** The evaluation engine. Fetches queries and metrics, interacts with the `model-registry` to get LLM responses, uses a judge model for scoring, and stores results in the database.
*   **`model-registry`:** Manages LLM providers and models. Provides a unified interface for generating text completions via adapters.
*   **`visualization-service`:** Backend API and React frontend for the dashboard. Queries the database to aggregate and present evaluation data.
*   **`postgres`:** Shared PostgreSQL database with `pgvector` extension, storing all persistent data except embeddings managed within item-storage.
*   **`migrations`:** Alembic setup for managing database schema evolution.

### Solution Diagram (Mermaid)

```mermaid
graph TD
    User -->|API Request| GW(main-app API Gateway :8000)

    subgraph Panopticon System
        GW -->|Forward Request| ISQ(item-storage-queries :8001)
        GW -->|Forward Request| ISM(item-storage-metrics :8002)
        GW -->|Forward Request| Judge(judge-service :8003)
        GW -->|Forward Request| MR(model-registry :8005)
        GW -->|Forward Request| Viz(visualization-service :8004)

        ISQ -->|Store/Fetch Queries| DB[(Postgres DB :5432)]
        ISM -->|Store/Fetch Metrics| DB
        Judge -->|Fetch Queries| ISQ
        Judge -->|Fetch Metrics| ISM
        Judge -->|Run Query/Evaluate| MR
        Judge -->|Store Results| DB
        MR -->|Store/Fetch Models/Providers| DB
        MR -->|External API Call| LLMAPI[External LLM APIs]
        Viz -->|Fetch Aggregated Data| DB
        Viz -->|Fetch Model Info| Judge # Or directly to MR? Check code - Fetching via Judge is safer decoupling
    end

    User -->|View Dashboard| Viz

    style GW fill:#f9f,stroke:#333,stroke-width:2px
    style DB fill:#ccf,stroke:#333,stroke-width:2px
```

*(This diagram shows the primary request flows. GitHub automatically renders Mermaid diagrams).*

## 🚀 Getting Started

### Prerequisites

*   **Docker:** [Install Docker](https://docs.docker.com/get-docker/)
*   **Docker Compose:** Usually included with Docker Desktop.
*   **Git:** To clone the repository.
*   **Python 3.12+:** (Optional, for local development or running scripts)
*   **LLM API Keys:** Obtain API keys for the providers you want to use (OpenAI, Google Gemini, Anthropic, etc.).
*   **`curl`:** Command-line tool for making HTTP requests.

### Setup Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/panopticon.git # Replace with your repo URL
    cd panopticon
    ```

2.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file:
        *   **Set a secure `API_KEY`** for internal service communication and external access. **Replace `your_api_key_here` below with this value.**
        *   Add your LLM API keys (`LITELLM_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, etc.). `LITELLM_API_KEY` is often used for OpenAI by default in LiteLLM, but check provider configs.
        *   Review `DATABASE_URL` and other PostgreSQL settings if you're not using the default Docker setup.

3.  **Build and Run with Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```
    *   `--build`: Forces Docker to build the images based on the Dockerfiles.
    *   `-d`: Runs the containers in detached mode (in the background).

4.  **Verify Services:**
    *   Check container logs: `docker-compose logs -f` (Press `Ctrl+C` to exit)
    *   Wait for services to become healthy (check `docker-compose ps`). Health checks are configured.
    *   Test the main API gateway:
        ```bash
        curl -H "X-API-Key: dev_api_key_for_testing" http://localhost:8000/health
        ```
    *   Test individual services (e.g., `curl http://localhost:8001/health`).

5.  **Access the Dashboard:**
    *   Open your web browser and navigate to `http://localhost:8004`.

## 📡 API Endpoints Overview

The primary way to interact with Panopticon is through the `main-app` API gateway running on port `8000`. An API key (`X-API-Key` header) matching the one in your `.env` file is required for most endpoints.

*   **`GET /`**: Basic info about the Panopticon system.
*   **`GET /health`**: Health check for the API gateway.
*   **`GET /api/services`**: Lists the available backend services and their primary endpoints.
*   **`POST /api/queries`**: Store a new query (prompt). (Targets `item-storage-queries`).
*   **`GET /api/queries/...`**: Retrieve or search queries. (Targets `item-storage-queries`).
*   **`POST /api/metrics`**: Store a new evaluation metric (prompt). (Targets `item-storage-metrics`).
*   **`GET /api/metrics/...`**: Retrieve or search metrics. (Targets `item-storage-metrics`).
*   **`POST /api/judge/evaluate/query`**: Evaluate a single query against a model using specified metrics. (Targets `judge-service`).
*   **`POST /api/judge/evaluate/theme`**: Evaluate all queries of a specific theme against a model. (Targets `judge-service`).
*   **`GET /api/judge/results`**: Get detailed evaluation results stored by the judge service.
*   **`GET /api/models`**: List models registered in the `model-registry`.
*   **`GET /api/providers`**: List LLM providers registered in the `model-registry`.
*   **`POST /api/completion`**: Directly generate text using a registered model (via `model-registry`).
*   **`GET /api/dashboard/...`**: Endpoints consumed by the visualization frontend to get aggregated data, timelines, comparisons, etc. (Targets `visualization-service`).

*Refer to the OpenAPI documentation available at `/api/docs` on the running `main-app` (http://localhost:8000/api/docs) for detailed request/response schemas.*

## 🚶 User Journey: Evaluating LLMs

Here’s a typical workflow for using Panopticon:

1.  **Define Your Queries:**
    *   Identify the tasks or prompts you want to evaluate (e.g., "Summarize the following text...", "Write Python code to...", "Explain this concept...").
    *   Group related queries under a `theme` (e.g., "summarization", "coding", "qa").
    *   **Action:** Send `POST` requests to `/api/queries` for each query:
        ```bash
        curl -X POST http://localhost:8000/api/queries \
             -H "Content-Type: application/json" \
             -H "X-API-Key: dev_api_key_for_testing" \
             -d '{
                   "item": "Summarize the provided article about renewable energy trends.",
                   "type": "summarization",
                   "metadata": { "source": "tech_crunch_article_123", "difficulty": "medium" }
                 }'
        ```
    *   **Best Practice:** Use consistent and descriptive themes. Add relevant metadata for later filtering.

2.  **Define Your Evaluation Metrics (Judge Prompts):**
    *   Decide how you want to score the LLM's responses. Create prompts for the *judge* model.
    *   **Action:** Send `POST` requests to `/api/metrics` for each evaluation criterion:
        ```bash
        curl -X POST http://localhost:8000/api/metrics \
             -H "Content-Type: application/json" \
             -H "X-API-Key: dev_api_key_for_testing" \
             -d '{
                   "item": "Evaluate the summary based on conciseness (1-10) and factual accuracy compared to the original text (1-10). Respond with ONLY a single score from 1 to 10, averaging the two criteria.",
                   "type": "summary_quality",
                   "metadata": { "version": "1.1", "author": "eval_team" }
                 }'
        ```
    *   **Best Practice:** Write clear, objective evaluation prompts for the judge. Ensure the prompt asks for a specific output format (like a single number). Keep track of the `id` returned in the response – you'll need it for evaluation.

3.  **Run Evaluations:**
    *   Choose the model(s) you want to test (e.g., `gpt-4o`, `claude-3-opus-20240229`) and the evaluation metric IDs from step 2.
    *   **Action (Single Query):** Send a `POST` request to `/api/judge/evaluate/query`:
        ```bash
        curl -X POST http://localhost:8000/api/judge/evaluate/query \
             -H "Content-Type: application/json" \
             -H "X-API-Key: dev_api_key_for_testing" \
             -d '{
                   "query": "Summarize the provided article about renewable energy trends.",
                   "model_id": "gpt-4o",
                   "theme": "summarization",
                   "evaluation_prompt_ids": ["<metric_id_from_step_2>"],
                   "judge_model": "gpt-4"
                 }'
        ```
    *   **Action (Entire Theme):** Send a `POST` request to `/api/judge/evaluate/theme`:
        ```bash
        curl -X POST http://localhost:8000/api/judge/evaluate/theme \
             -H "Content-Type: application/json" \
             -H "X-API-Key: dev_api_key_for_testing" \
             -d '{
                   "theme": "summarization",
                   "model_id": "claude-3-opus-20240229",
                   "evaluation_prompt_ids": ["<metric_id_from_step_2>"],
                   "judge_model": "gpt-4",
                   "limit": 50
                 }'
        ```
    *   **Recommendation:** Start with evaluating single queries or small theme batches (`limit`) to ensure prompts and metrics work as expected before running large-scale evaluations. Repeat for different models.

4.  **Analyze Results:**
    *   **Action:** Open the Visualization Dashboard at `http://localhost:8004`.
    *   Explore the different pages:
        *   **Dashboard:** Overall summary statistics and trends.
        *   **Model Comparison:** Side-by-side performance using bar and radar charts.
        *   **Theme Analysis:** Heatmap showing model strengths/weaknesses across themes.
        *   **Detailed Results:** A filterable table view of individual evaluation records.
    *   **Action (Programmatic):** Use `GET /api/dashboard/results` with filters to fetch raw data for custom analysis (using `curl` or another HTTP client). Example:
        ```bash
        # Get first 10 results for 'summarization' theme by model 'gpt-4o'
        curl -G http://localhost:8000/api/judge/results \
             -H "X-API-Key: dev_api_key_for_testing" \
             --data-urlencode "theme=summarization" \
             --data-urlencode "model_id=gpt-4o" \
             --data-urlencode "limit=10"
        ```
    *   **Best Practice:** Use the dashboard for high-level insights and trend spotting. Use the API or direct database queries for deep dives or specific statistical analysis.

## 💻 Technology Stack

*   **Backend:** Python, FastAPI
*   **Frontend:** React, Vite, Tailwind CSS, Chart.js, Plotly.js
*   **Database:** PostgreSQL, pgvector
*   **LLM Interaction:** LiteLLM, sentence-transformers
*   **Containerization:** Docker, Docker Compose
*   **Database Migrations:** Alembic
*   **Async:** `asyncio`, `aiohttp`, `asyncpg`

## 🤝 Contributing

Contributions are welcome! Please follow standard Forking Workflow:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -am 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Create a new Pull Request.

Please ensure your code follows the style guidelines (Black, Ruff, isort) and includes tests where applicable.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤔 Support & Questions

If you encounter issues or have questions, please file an issue on the GitHub repository.
---

*Happy Evaluating!* 🚀
```