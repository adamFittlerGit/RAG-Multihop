#!/bin/bash

python3 MyRetEval.py --stage_one_filename output/stage-one/rerankerA/llm-embedder--bge-reranker-base.json > results/stage-one/llm-embedder--bge-reranker-base.txt

python3 MyRetEval.py --stage_one_filename output/stage-one/rerankerB/llm-embedder--jina-reranker-v2-base-multilingual.json > results/stage-one/llm-embedder--jina-reranker-v2-base-multilingual.txt

python3 MyRetEval.py --stage_one_filename output/stage-one/rerankerC/llm-embedder--mxbai-rerank-base-v1.json > results/stage-one/llm-embedder--mxbai-rerank-base-v1.txt

python3 MyRetEval.py --stage_one_filename output/stage-one/rerankerA/all-mpnet-base-v2--bge-reranker-base.json > results/stage-one/all-mpnet-base-v2--bge-reranker-base.txt

python3 MyRetEval.py --stage_one_filename output/stage-one/rerankerB/all-mpnet-base-v2--jina-reranker-v2-base-multilingual.json > results/stage-one/all-mpnet-base-v2--jina-reranker-v2-base-multilingual.txt

python3 MyRetEval.py --stage_one_filename output/stage-one/rerankerC/all-mpnet-base-v2--mxbai-rerank-base-v1.json > results/stage-one/all-mpnet-base-v2--mxbai-rerank-base-v1.txt

python3 MyRetEval.py --stage_one_filename output/stage-one/rerankerA/instructor-base--bge-reranker-base.json > results/stage-one/instructor-base--bge-reranker-base.txt

python3 MyRetEval.py --stage_one_filename output/stage-one/rerankerB/instructor-base--jina-reranker-v2-base-multilingual.json > results/stage-one/instructor-base--jina-reranker-v2-base-multilingual.txt

python3 MyRetEval.py --stage_one_filename output/stage-one/rerankerC/instructor-base--mxbai-rerank-base-v1.json > results/stage-one/instructor-base--mxbai-rerank-base-v1.txt
