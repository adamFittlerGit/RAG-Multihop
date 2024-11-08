import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate import meteor_score
nltk.download("wordnet")
nltk.download('punkt_tab')
nltk.download('punkt')

from rouge_score import rouge_scorer

import matplotlib.pyplot as plt

from evaluate import load

import json, sys, os
from tqdm import tqdm
import re
from collections import Counter

# Function to get the correct answer
def get_gold(query_data, query):
  for q in query_data:
    if q['query'] == query:
      return q['answer']
  return ''

# Function to check if there is an intersection of words between two strings
def has_intersection(a, b):
  a_words = set(a.split())
  b_words = set(b.split())
  return len(a_words.intersection(b_words)) > 0

def get_idx(x,alist):
  for i,c in enumerate(alist):
    if c == x:
      return i
  return -1

def count_overlap(gold,pred):
  # Standardise by removing all non-alphanumeric characters.
  # The input should be lower cased. But to be safe ...
  g = gold.lower()
  p = pred.lower()
  cg = re.sub(r'[^A-Za-z0-9 ]+', '', g)
  cp = re.sub(r'[^A-Za-z0-9 ]+', '', p)
  gold_words = cg.split()
  pred_words = cp.split()
  glen = len(gold_words)
  plen = len(pred_words)

  # Somewhat destructive as it removes dupes, but is the only sensible way
  # to do it.
  #gold_words = list(set(gold_words))
  #pred_words = list(set(pred_words))
  cnt = 0
  for w in pred_words:
    rv = get_idx(w,gold_words)
    if rv != -1:
      cnt += 1
      v = gold_words.pop(rv)
  return cnt, glen, plen 

# Function to extract the answer from gold
def extract_answer(input_string):
  for str in [':', '<eos', '`', 'The answer is', '\n', '*']:
    input_string.replace(str, "")
  match = re.search(r'The answer to the question is "(.*?)"', input_string)
  return match.group(1) if match else input_string

def comp_metrics_new(pred_list, gold_list):
  bertscore = load("bertscore")
  bertscores = bertscore.compute(predictions=pred_list, references=gold_list, lang='en')
  bp_list = bertscores['precision']
  br_list = bertscores['recall']
  prec_list = []
  recall_list = []
  f1_list = []
  meteor_list = []
  rouge_list = []

  for gold, pred in zip(gold_list, pred_list):
    c, plen, glen = count_overlap(gold,pred)

    # Compute Precision Directly
    if plen == 0:
      precision = 0.0
    else:
      precision = float(c)/plen

    # Compute Recall Directly
    if glen == 0:
      recall = 0
    else:
      recall = float(c)/glen

    if precision == 0.0 and recall == 0.0:
      f1 = 0.0
    else:
      f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
      
    meteor = meteor_score.single_meteor_score(nltk.word_tokenize(gold), nltk.word_tokenize(pred))
    meteor_list.append(meteor)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(gold, pred)

    rouge_list.append(rouge['rougeL'].fmeasure)
    prec_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

  # You can modify this code easily to get the list of per query scores for all
  # three metrics.
  micro_prec = sum(prec_list)/len(prec_list)
  micro_recall = sum(recall_list)/len(recall_list)
  micro_f1 = sum(f1_list)/len(f1_list)
  micro_meteor = sum(meteor_list)/len(meteor_list) 
  micro_rouge = sum(rouge_list)/len(rouge_list)
  micro_bp = sum(bp_list)/len(bp_list)
  micro_br = sum(br_list)/len(br_list)
  
  results_list = [meteor_list, rouge_list, bp_list, br_list]

  return micro_prec, micro_recall, micro_f1, micro_meteor, micro_rouge, micro_bp, micro_br, results_list


# Function to calculate evaluation metrics
def comp_metrics(pred_list, gold_list):
  tp = sum(1 for pred, gold in zip(pred_list, gold_list) 
           if has_intersection(pred.lower(), gold.lower()))
  fp = sum(1 for pred, gold in zip(pred_list, gold_list) 
           if not has_intersection(pred.lower(), gold.lower()))
  fn = len(gold_list) - tp
  #print ('{} {} {}'.format(tp, fp, fn))
  #print (len(gold_list))
  #print (len(pred_list))
  precision = tp / (tp + fp) if tp + fp > 0 else 0
  recall = tp / (tp + fn) if tp + fn > 0 else 0
  f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
  meteor = meteor_score.single_meteor_score(gold_list, pred_list)
  bleu = sentence_bleu(gold_list, pred_list, smoothing_function=SmoothingFunction().method1)
  return precision, recall, f1, meteor, bleu

def run_evaluation(predictions, gold_labels, output_dir):
  # Read files
  with open(predictions, 'r') as fh:
    data = fh.read()
    doc_data = json.loads(data)

  #with open('dataset/MultiHopRAG.json', 'r') as file:
  with open(gold_labels, 'r') as fh:
    data = fh.read()
    query_data = json.loads(data)

  # Initialize dictionary to save lists of predictions and gold standards 
  # for each question_type
  type_data = {}
  overall_pred_list = []
  overall_gold_list = []

  #print(doc_data)
  # Main loop, iterate through document data
  for d in tqdm(doc_data):
    model_answer = d['model_answer']
    model_answer = extract_answer(model_answer)
    gold = get_gold(query_data,d['query'])
    if gold:
      question_type = d['question_type']
      if question_type not in type_data:
        type_data[question_type] = {'pred_list': [], 'gold_list': []}
      type_data[question_type]['pred_list'].append(model_answer)
      type_data[question_type]['gold_list'].append(gold)
      overall_pred_list.append(model_answer)
      overall_gold_list.append(gold)

  # Output evaluation data for each question_type
  for question_type, data in type_data.items():
    precision, recall, f1, meteor, rouge, bp, br, results_list = comp_metrics_new(data['pred_list'], data['gold_list'])
    print(f"Question Type: {question_type}")
    print(f" Precision: {precision:.2f}")
    print(f" Recall: {recall:.2f}")
    print(f" F1 Score: {f1:.2f}")
    print(f" Meteor Score: {meteor:.2f}")
    print(f" Rouge F1 Score: {rouge:.2f}")
    print(f" Bert Precision Score: {bp:.2f}")
    print(f" Bert Recall Score: {br:.2f}")
    # Print the scores with proper formatting
    print()
    
    data = results_list

    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = ax.boxplot(data, patch_artist = True, notch ='True', vert = 0)

    colors = ['#FF0000', '#00FF00', '#0000FF', '#800080']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',linewidth = 1.5, linestyle =":")
    
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)
    
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)
                    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',color ='#e7298a',alpha = 0.5)
                                                
    # x-axis labels
    ax.set_yticklabels(['Meteor', 'Rouge-L F1', 'Bert P', 'Bert R'])

    # Adding title 
    plt.title(f"{question_type} per query results")

    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
                                                

    # show plot
    plt.savefig(os.path.join(output_dir, f'{question_type}.png'))


  # Calculate overall evaluation metrics
  overall_precision, overall_recall, overall_f1, overall_meteor, overall_rouge, overall_bp, overall_br, overall_list = comp_metrics_new(overall_pred_list, 
                                                                   overall_gold_list)
  print(f"Overall Metrics:")
  print(f" Precision: {overall_precision:.2f}")
  print(f" Recall: {overall_recall:.2f}")
  print(f" F1 Score: {overall_f1:.2f}")
  print(f" Meteor Score: {overall_meteor:.2f}")
  print(f" Rouge F1 Score: {overall_rouge:.2f}")
  print(f" Bert Precision Score: {overall_bp:.2f}")
  print(f" Bert Recall Score: {overall_br:.2f}")


  data = overall_list

  fig = plt.figure(figsize =(10, 7))
  ax = fig.add_subplot(111)

  # Creating axes instance
  bp = ax.boxplot(data, patch_artist = True, notch ='True', vert = 0)

  colors = ['#FF0000', '#00FF00', '#0000FF', '#800080']

  for patch, color in zip(bp['boxes'], colors):
      patch.set_facecolor(color)

  # changing color and linewidth of
  # whiskers
  for whisker in bp['whiskers']:                                                                                                                                                                                       whisker.set(color ='#8B008B',linewidth = 1.5, linestyle =":")

  # changing color and linewidth of
  # caps
  for cap in bp['caps']:
      cap.set(color ='#8B008B', linewidth = 2)
  
  for median in bp['medians']:
      median.set(color ='red', linewidth = 3)
  
  for flier in bp['fliers']:                                                                                  
      flier.set(marker ='D',color ='#e7298a',alpha = 0.5)

  # x-axis labels
  ax.set_yticklabels(['Meteor', 'Rouge-L F1', 'Bert P', 'Bert R'])

  plt.title(f"overall per query result")
  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()
  plt.savefig(os.path.join(output_dir, 'overall.png'))

if __name__ == '__main__':
    # prediction_file = 'output/llama2.json'
    prediction_file = sys.argv[1]
    output_dir = sys.argv[2]
    gold_labels = 'data/rag.json'
    run_evaluation(prediction_file, gold_labels, output_dir)

