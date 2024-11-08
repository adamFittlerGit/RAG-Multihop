#!/bin/bash

yes | python3 RAGA.py --stage_one "output/instructor-base--jina-reranker-v2-base-multilingual.json"
yes | python3 RAGA.py --stage_one "output/all-mpnet-base-v2--mxbai-rerank-base-v1.json"
yes | python3 RAGB.py --stage_one "output/instructor-base--jina-reranker-v2-base-multilingual.json"
yes | python3 RAGB.py --stage_one "output/all-mpnet-base-v2--mxbai-rerank-base-v1.json"
yes | python3 RAGC.py --stage_one "output/instructor-base--jina-reranker-v2-base-multilingual.json"
yes | python3 RAGC.py --stage_one "output/all-mpnet-base-v2--mxbai-rerank-base-v1.json"
yes | python3 RAGD.py --stage_one "output/instructor-base--jina-reranker-v2-base-multilingual.json"
yes | python3 RAGD.py --stage_one "output/all-mpnet-base-v2--mxbai-rerank-base-v1.json"
