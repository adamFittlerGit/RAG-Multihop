#!/bin/bash

yes | python3 rerankerA.py --rank_model_name "BAAI/llm-embedder"
yes | python3 rerankerA.py --rank_model_name "sentence-transformers/all-mpnet-base-v2"
yes | python3 rerankerA.py --rank_model_name "hkunlp/instructor-base"
yes | python3 rerankerB.py --rank_model_name "BAAI/llm-embedder"
yes | python3 rerankerB.py --rank_model_name "sentence-transformers/all-mpnet-base-v2"
yes | python3 rerankerB.py --rank_model_name "hkunlp/instructor-base"
yes | python3 rerankerC.py --rank_model_name "BAAI/llm-embedder"
yes | python3 rerankerC.py --rank_model_name "sentence-transformers/all-mpnet-base-v2"
yes | python3 rerankerC.py --rank_model_name "hkunlp/instructor-base"
