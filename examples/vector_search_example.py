"""
Example script demonstrating vector search capabilities.
"""
import asyncio
from datetime import datetime, timedelta

from llm_eval.core.models import Prompt, PromptCategory, LLMProvider, LLMResponse
from llm_eval.services.storage_service.enhanced_storage import EnhancedStorageService


async def main():
    # Initialize enhanced storage service
    storage = EnhancedStorageService(
        # PostgreSQL settings
        pg_host="localhost",
        pg_port=5432,
        pg_user="llmeval",
        pg_password="llmeval",
        pg_database="llmeval",
        # Use in-memory mode for demo
        persist_directory="./vector_db_demo"
    )
    
    await storage.initialize()
    
    print("Creating sample prompts and responses...")
    
    # Create sample prompts
    prompt1 = Prompt(
        text="Explain quantum computing to a high school student",
        category=PromptCategory.SCIENCE_TECHNOLOGY,
        tags=["quantum", "education"]
    )
    
    prompt2 = Prompt(
        text="What are the ethical implications of artificial intelligence?",
        category=PromptCategory.PHILOSOPHY_ETHICS,
        tags=["ai", "ethics"]
    )
    
    # Store prompts
    prompt1_result = await storage.store_prompt(prompt1)
    prompt2_result = await storage.store_prompt(prompt2)
    
    if prompt1_result.is_err or prompt2_result.is_err:
        print("Error storing prompts")
        return
    
    stored_prompt1 = prompt1_result.unwrap()
    stored_prompt2 = prompt2_result.unwrap()
    
    # Create sample responses
    response1_gpt = LLMResponse(
        prompt_id=stored_prompt1.id,
        prompt_text=stored_prompt1.text,
        model_name="gpt-4-0125-preview",
        provider=LLMProvider.OPENAI,
        response_text="Quantum computing uses quantum bits or qubits to perform calculations. "
                     "Unlike regular bits that are either 0 or 1, qubits can exist in a superposition "
                     "of both states simultaneously, allowing quantum computers to solve certain "
                     "problems much faster than classical computers."
    )
    
    response1_claude = LLMResponse(
        prompt_id=stored_prompt1.id,
        prompt_text=stored_prompt1.text,
        model_name="claude-3-7-sonnet-20250219",
        provider=LLMProvider.ANTHROPIC,
        response_text="Imagine regular computers as working with coins that are either heads or tails. "
                     "Quantum computers use special 'quantum coins' that can be heads, tails, or spinning "
                     "in between - representing both states at once through a property called superposition. "
                     "This allows quantum computers to explore multiple possibilities simultaneously."
    )
    
    response2_gpt = LLMResponse(
        prompt_id=stored_prompt2.id,
        prompt_text=stored_prompt2.text,
        model_name="gpt-4-0125-preview",
        provider=LLMProvider.OPENAI,
        response_text="AI raises ethical concerns like privacy, bias, job displacement, and autonomous "
                     "decision-making. We need responsible development practices, transparency, and "
                     "regulations to ensure AI benefits humanity while minimizing potential harms."
    )
    
    # Store responses
    await storage.store_response(response1_gpt)
    await storage.store_response(response1_claude)
    await storage.store_response(response2_gpt)
    
    print("Stored prompts and responses. Now let's try some vector search...")
    
    # Text-based semantic search
    print("\n1. Searching for content about 'particles in quantum physics':")
    results = await storage.query_embeddings_by_text(
        query_text="particles in quantum physics",
        limit=2
    )
    
    if results.is_ok:
        for i, result in enumerate(results.unwrap()):
            print(f"\nResult {i+1}:")
            print(f"Score: {result.get('similarity', 'N/A')}")
            print(f"Text: {result.get('text', '')[:100]}...")
    
    # Search with metadata filter
    print("\n2. Searching for content about 'AI ethical concerns' from OpenAI models only:")
    results = await storage.query_embeddings_by_text(
        query_text="AI ethical concerns",
        filter_metadata={"provider": "openai"},
        limit=2
    )
    
    if results.is_ok:
        for i, result in enumerate(results.unwrap()):
            print(f"\nResult {i+1}:")
            print(f"Model: {result.get('metadata', {}).get('model_name', 'unknown')}")
            print(f"Text: {result.get('text', '')[:100]}...")
    
    # Compare models semantically
    print("\n3. Comparing model responses semantically:")
    comparison = await storage.compare_responses_semantically(
        prompt_id=stored_prompt1.id,
        model_names=["gpt-4-0125-preview", "claude-3-7-sonnet-20250219"]
    )
    
    if comparison.is_ok:
        result = comparison.unwrap()
        for pair, scores in result["similarity_scores"].items():
            print(f"Similarity between {pair}: {scores['average']:.4f}")
    
    # Semantic drift detection requires responses from different time periods
    # For demo, we'll create some artificial responses with different timestamps
    
    print("\nExample complete!")
    
    # Clean up (optional)
    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
