import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid
import litellm
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class EvaluationResult(Base):
    __tablename__ = 'evaluation_results'
    
    id = Column(String, primary_key=True)
    query_id = Column(String)
    model_id = Column(String)
    theme = Column(String)
    evaluation_prompt_id = Column(String)
    score = Column(Float)
    judge_model = Column(String)
    timestamp = Column(DateTime)
    result_metadata = Column(JSON)

class JudgeStorage:
    def __init__(
        self, 
        persist_directory: str = "./judge_data",
        postgres_url: str = "postgresql://postgres:postgres@postgres:5432/panopticon"
    ):
        # Initialize ChromaDB for storing LLM outputs
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name="llm_outputs",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize PostgreSQL for storing evaluation results
        self.engine = create_engine(postgres_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    async def run_query_with_llm(
        self, 
        query: str, 
        model_id: str,
        theme: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run a query through LiteLLM and store the output
        """
        # Run query through LiteLLM
        response = await litellm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": query}]
        )
        
        output_text = response.choices[0].message.content
        output_id = str(uuid.uuid4())
        
        # Prepare metadata
        output_metadata = {
            "model_id": model_id,
            "theme": theme,
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        # Store in ChromaDB
        self.collection.add(
            documents=[output_text],
            metadatas=[output_metadata],
            ids=[output_id]
        )
        
        return {
            "id": output_id,
            "output": output_text,
            "metadata": output_metadata
        }

    async def evaluate_output(
        self,
        query: str,
        output: str,
        evaluation_prompt: str,
        evaluation_prompt_id: str,
        model_id: str,
        theme: str,
        judge_model: str = "gpt-4",
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an LLM output using a judge LLM
        """
        evaluation_prompt_template = """
        You are an expert evaluator. Your task is to evaluate an AI's response based on specific criteria.
        
        Query: {query}
        AI Response: {output}
        
        Evaluation Criteria: {evaluation_prompt}
        
        Please provide a score from 1-10 where:
        1 = Completely fails to meet the criteria
        10 = Perfectly meets the criteria
        
        Respond with ONLY a number between 1 and 10.
        """
        
        formatted_prompt = evaluation_prompt_template.format(
            query=query,
            output=output,
            evaluation_prompt=evaluation_prompt
        )
        
        # Get evaluation from judge LLM
        response = await litellm.acompletion(
            model=judge_model,
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        
        score = float(response.choices[0].message.content.strip())
        
        # Store evaluation result in PostgreSQL
        result = EvaluationResult(
            id=str(uuid.uuid4()),
            query_id=metadata.get("query_id") if metadata else None,
            model_id=model_id,
            theme=theme,
            evaluation_prompt_id=evaluation_prompt_id,
            score=score,
            judge_model=judge_model,
            timestamp=datetime.utcnow(),
            result_metadata=metadata
        )
        
        self.session.add(result)
        self.session.commit()
        
        return {
            "score": score,
            "metadata": {
                "query": query,
                "output": output,
                "evaluation_prompt": evaluation_prompt,
                "model_id": model_id,
                "theme": theme,
                "judge_model": judge_model,
                **(metadata or {})
            }
        }

    def get_outputs_by_theme_and_model(
        self, 
        theme: str, 
        model_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Retrieve LLM outputs by theme and model
        """
        results = self.collection.query(
            query_texts=[""],
            where={"theme": theme, "model_id": model_id},
            n_results=limit
        )
        
        outputs = []
        for i in range(len(results['ids'][0])):
            outputs.append({
                "id": results['ids'][0][i],
                "output": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        
        return outputs

    def get_evaluation_results(
        self,
        theme: Optional[str] = None,
        model_id: Optional[str] = None,
        evaluation_prompt_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Query evaluation results from PostgreSQL
        """
        query = self.session.query(EvaluationResult)
        
        if theme:
            query = query.filter(EvaluationResult.theme == theme)
        if model_id:
            query = query.filter(EvaluationResult.model_id == model_id)
        if evaluation_prompt_id:
            query = query.filter(EvaluationResult.evaluation_prompt_id == evaluation_prompt_id)
            
        results = query.limit(limit).all()
        
        return [
            {
                "id": result.id,
                "query_id": result.query_id,
                "model_id": result.model_id,
                "theme": result.theme,
                "evaluation_prompt_id": result.evaluation_prompt_id,
                "score": result.score,
                "judge_model": result.judge_model,
                "timestamp": result.timestamp.isoformat(),
                "metadata": result.result_metadata
            }
            for result in results
        ] 