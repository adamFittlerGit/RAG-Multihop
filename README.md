---

# Performance Evaluation of RAG Models

## Overview
This project provides a comprehensive evaluation of ranking and reranking models within a Retrieval-Augmented Generation (RAG) framework, focusing on identifying optimal configurations for information retrieval and natural language generation. It benchmarks three rankers and three rerankers, integrated with four text generators, across multiple performance metrics.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Experimental Results](#experimental-results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
Modern information retrieval systems require precise ranking models to order search results or generated text based on relevance. This project explores the impact of various ranker-reranker combinations on the quality and relevance of generated text, particularly in large language models (LLMs) designed for contextual and instructional generation. Three rankers and three rerankers are tested in conjunction with four state-of-the-art text generators to analyze combinations that yield optimal or suboptimal results.

## Methodology
The study evaluates three rankers and three rerankers:
- **Rankers**:
  - Ranker A: llm-embedder
  - Ranker B: instructor-base
  - Ranker C: all-mpnet-base-v2
- **Rerankers**:
  - Reranker A: bge-reranker-base
  - Reranker B: jina-reranker-v2
  - Reranker C: mxbai-rerank-base-v1
- **Text Generators**:
  - Generator A: Gemma-2-2b
  - Generator B: Llama-2-7b
  - Generator C: Mistral-7B
  - Generator D: Qwen2.5-7B

Each combination of rankers and rerankers is tested on Hits@10, Hits@4, MAP@10, and MRR@10 metrics, followed by evaluation of generator performance with Precision, Recall, F1, METEOR, ROUGE-L F1, and BERT Score.

## Experimental Results
The project identifies that:
- The **B-B-D combination** (Ranker B with Reranker B and Qwen2.5-7B generator) achieves the highest scores, particularly in BERT Precision (87%) and Recall (91%), signifying strong semantic alignment.
- The **C-C-C combination** (Ranker C, Reranker C, Mistral-7B generator) shows weaker performance, highlighting challenges with token alignment and coherence.

Metrics indicate that BERT-based evaluations provide better semantic alignment insights, while METEOR and ROUGE scores emphasize structural alignment.

## Conclusion
The study reveals that optimal RAG combinations can enhance contextual relevance and coherence in generated text. Future studies may explore more diverse query types, larger models, and human evaluation for richer insights.

## References
This project references models and methodologies accessible via the Hugging Face platform and common IR/NLP metrics for detailed performance analysis.

---