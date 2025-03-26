"""
Test script for the direct Gemini client.
"""
import asyncio
import os
from direct_gemini_client import DirectGeminiClient

async def test_client():
    print("Testing DirectGeminiClient with Gemini API")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not found")
        return
    
    print(f"Using API key: {api_key[:5]}...")
    
    try:
        # Initialize client
        client = DirectGeminiClient(api_key=api_key)
        print("Client initialized successfully")
        
        # Test with a simple query
        model_id = "gemini-2.5-pro-exp-03-25"  # The model that's having issues
        print(f"Testing with model: {model_id}")
        
        messages = [{"role": "user", "content": "Hello, what is quantum computing?"}]
        
        # Generate content
        response = await client.generate_content(
            model_id=model_id,
            messages=messages,
            temperature=0.7
        )
        
        print("\nSUCCESS! Got response:")
        print(f"Content: {response['content'][:200]}...")
        print(f"Usage: {response['usage']}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_client())
