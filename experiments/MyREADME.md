#### Running my code
## Environment Setup
In order to run the stage one and stage two combinations simply activate the conda or venv environment with
and install the packages in requirements.txt using `pip install -r requirements.txt`.  When using some of the models
there may be an error with the transfomers library if this occurs `pip uninstall tranformers` followed by
`pip install transformers --no-cache` seemed to fix the problem for me.  

## Running Stage One
Running stage one requires running the script `bash run_stage_one.sh`, the outputs will be saved to `output/stage-one/reranker[A-B]`
in order to keep the outputs organised, the outputs will be names in the format `[ranker--reranker].json`.

In order to evaluate stage one run `bash eval_stage_one.sh` in order to save the results for each to text files located at 
`results/stage_one/[ranker--reranker.txt]`, next run `bash concat_results.sh` to create save all results to `results/stage_one/all_files_content.txt`

## Running Stage two
Running stage two consists of running the script `bash run_stage_two.sh` saving the outputs to `output/stage-two/RAG[A-D]/[ranker--reranker--generator].json`
for each of the models for easy organisation.  Each model is run with the best and worst scoring retrieval stage in order to test the effect on the outcomes.

In order to evalaute the results simply run `bash eval_stage_two.sh` and the results will be saved to `results/stage_two/s1-[best/worst]-[model_name]`
here all of the per query result graphs are saved as well as results.txt for each outlining all of the resultant metrics for analysis.

