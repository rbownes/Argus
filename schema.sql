-- LLM Evaluation System Database Schema

-- Run this script to initialize the necessary database tables

-- Evaluation runs table
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id UUID PRIMARY KEY,
    models JSONB NOT NULL,
    themes JSONB NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Model responses table
CREATE TABLE IF NOT EXISTS model_responses (
    id UUID PRIMARY KEY,
    prompt_id UUID NOT NULL,
    model_provider VARCHAR(50) NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    latency_ms INTEGER,
    tokens_used INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_model_responses_model ON model_responses(model_provider, model_id);
CREATE INDEX IF NOT EXISTS idx_model_responses_timestamp ON model_responses(timestamp);

-- Evaluation results table
CREATE TABLE IF NOT EXISTS evaluation_results (
    id UUID PRIMARY KEY,
    response_id UUID NOT NULL REFERENCES model_responses(id),
    run_id UUID NOT NULL REFERENCES evaluation_runs(id),
    evaluator_id VARCHAR(100) NOT NULL,
    scores JSONB NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_evaluation_results_run ON evaluation_results(run_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_response ON evaluation_results(response_id);

-- Prompts table to store reusable prompts
CREATE TABLE IF NOT EXISTS prompts (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    theme VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create index for theme-based queries
CREATE INDEX IF NOT EXISTS idx_prompts_theme ON prompts(theme);
