#!/bin/bash

# Generators using the best stage one model
python3 MyRAGEval.py output/stage-two/RAGA/instructor-base--jina-reranker-v2-base-multilingual--llama2.json results/stage_two/s1-best-llama2 > results/stage_two/s1-best-llama2/results.txt
python3 MyRAGEval.py output/stage-two/RAGB/instructor-base--jina-reranker-v2-base-multilingual--qwen.json results/stage_two/s1-best-qwen > results/stage_two/s1-best-qwen/results.txt
python3 MyRAGEval.py output/stage-two/RAGC/instructor-base--jina-reranker-v2-base-multilingual--minstral.json results/stage_two/s1-best-min > results/stage_two/s1-best-min/results.txt
python3 MyRAGEval.py output/stage-two/RAGD/instructor-base--jina-reranker-v2-base-multilingual--gemma.json results/stage_two/s1-best-gemma > results/stage_two/s1-best-gemma/results.txt
# Generators using worst stage one model
python3 MyRAGEval.py output/stage-two/RAGA/all-mpnet-base-v2--mxbai-rerank-base-v1--llama2.json results/stage_two/s1-worst-llama2 > results/stage_two/s1-worst-llama2/results.txt
python3 MyRAGEval.py output/stage-two/RAGB/all-mpnet-base-v2--mxbai-rerank-base-v1--qwen.json results/stage_two/s1-worst-qwen > results/stage_two/s1-worst-qwen/results.txt
python3 MyRAGEval.py output/stage-two/RAGC/all-mpnet-base-v2--mxbai-rerank-base-v1--minstral.json results/stage_two/s1-worst-min > results/stage_two/s1-worst-min/results.txt
python3 MyRAGEval.py output/stage-two/RAGD/all-mpnet-base-v2--mxbai-rerank-base-v1--gemma.json results/stage_two/s1-worst-gemma > results/stage_two/s1-worst-gemma/results.txt

