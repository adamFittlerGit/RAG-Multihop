import argparse
import json, os
import torch
from typing import Any, Generator, List, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig, pipeline
from transformers.modelcard import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
torch.set_default_dtype(torch.float16)

from llama_index.schema import Document

from huggingface_hub import login

# Log in to Hugging Face
login("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")


def save_list_to_json(lst, filename):
  """ Save Files """
  with open(filename, 'w') as file:
    json.dump(lst, file)

def wr_dict(filename,dic):
  """ Write Files """
  try:
    if not os.path.isfile(filename):
      data = []
      data.append(dic)
      with open(filename, 'w') as f:
        json.dump(data, f)
    else:
      with open(filename, 'r') as f:
        data = json.load(f)
        data.append(dic)
      with open(filename, 'w') as f:
          json.dump(data, f)
  except Exception as e:
    print("Save Error:", str(e))
  return

def rm_file(file_path):
  """ Delete Files """
  if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File {file_path} removed successfully.")

def _depth_first_yield(json_data: Any, levels_back: int, collapse_length:
                       Optional[int], path: List[str], ensure_ascii: bool = False,
                      ) -> Generator[str, None, None]:
  """ Do depth first yield of all of the leaf nodes of a JSON.
      Combines keys in the JSON tree using spaces.
      If levels_back is set to 0, prints all levels.
      If collapse_length is not None and the json_data is <= that number
      of characters, then we collapse it into one line.
  """
  if isinstance(json_data, (dict, list)):
    # only try to collapse if we're not at a leaf node
    json_str = json.dumps(json_data, ensure_ascii=ensure_ascii)
    if collapse_length is not None and len(json_str) <= collapse_length:
      new_path = path[-levels_back:]
      new_path.append(json_str)
      yield " ".join(new_path)
      return
    elif isinstance(json_data, dict):
      for key, value in json_data.items():
        new_path = path[:]
        new_path.append(key)
        yield from _depth_first_yield(value, levels_back, collapse_length, new_path)
    elif isinstance(json_data, list):
      for _, value in enumerate(json_data):
        yield from _depth_first_yield(value, levels_back, collapse_length, path)
    else:
      new_path = path[-levels_back:]
      new_path.append(str(json_data))
      yield " ".join(new_path)


class JSONReader():
  """JSON reader.
     Reads JSON documents with options to help suss out relationships between nodes.
  """
  def __init__(self, is_jsonl: Optional[bool] = False,) -> None:
    """Initialize with arguments."""
    super().__init__()
    self.is_jsonl = is_jsonl

  def load_data(self, input_file: str) -> List[Document]:
    """Load data from the input file."""
    documents = []
    with open(input_file, 'r') as file:
      load_data = json.load(file)
    for data in load_data:
      metadata = {"title": data['title'],
                  "published_at": data['published_at'],
                  "source":data['source']}
      documents.append(Document(text=data['body'], metadata=metadata))
    return documents


def run_query(tokenizer, model, messages, temperature=0.0, max_new_tokens=512, **kwargs,):
  messages = [
    {"role": "user", "content": messages},
  ]
  input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

  outputs = model.generate(**input_ids, max_new_tokens=256)

  res = tokenizer.decode(outputs[0])

  start_tag = "<end_of_turn>"
  end_tag = "<end_of_turn>"

  # Find the start and end indices of the tags
  start_index = res.find(start_tag) + len(start_tag)
  end_index = res.find(end_tag, start_index)

  # Extract the text between the tags
  result = res[start_index:end_index].strip()

  print(result)

  return result


def initialise_and_run_model(save_name, input_stage_1, model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
      model_name, device_map="auto", torch_dtype=torch.bfloat16,
  )

  # You can change this instruction prompt if you want, but be careful. This one
  # is carefully tested and if you do not return information as defined here,
  # evaluation will fail.
  prefix = """Below is a question followed by some context from different sources.
            Please answer the question based on the context.
            The answer to the question is a word or entity (respond with a single word or entity).
            If the provided information is insufficient to answer the question,
            respond 'Insufficient Information'.
            Answer directly without explanation. Do not provide the * character in tyour response or 
            ``` or the phrase answer:, just the answer to allow for your answer to be directly compared to the gold label"""

  print('Loading Stage 1 Ranking')
  with open(input_stage_1, 'r') as file:
    doc_data = json.load(file)

  print('Remove saved file if exists.')
  rm_file(save_name)

  save_list = []
  for d in tqdm(doc_data):
    retrieval_list = d['retrieval_list']
    context = '--------------'.join(e['text'] for e in retrieval_list)
    prompt = f"{prefix}\n\nQuestion:{d['query']}\n\nContext:\n\n{context}"
    response = run_query(tokenizer, model, prompt)
    #print(response)
    save = {}
    save['query'] = d['query']
    save['prompt'] = prompt
    save['model_answer'] = response
    save['gold_answer'] = d['answer']
    save['question_type'] = d['question_type']
    #print(save)
    save_list.append(save)

  # Save Results
  print ('Query processing completed. Saving the results.')
  save_list_to_json(save_list,save_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--stage_one", help="Stage one results used for generation")

  args = parser.parse_args()

  model_name = "google/gemma-2-2b-it"
  input_stage_1 = args.stage_one
  output_file = f"output/stage-two/RAGD/{(input_stage_1.split('/')[-1]).split('.')[0]}--gemma.json"


  initialise_and_run_model(output_file, input_stage_1, model_name)