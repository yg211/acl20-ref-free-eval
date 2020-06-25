# SUPERT: Unsupervised Multi-Document Summarization Evaluation & Generation

This project includes the source code for the paper [**SUPERT: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization**](https://arxiv.org/abs/2005.03724), to appear at ACL 2020.

**Highlighted Features**

* *Unsupervised evaluation metrics*: Measure multi-document summaries without using human-written reference summaries
* *Unsupervised multi-document summarizer*: Using the unsupervised evaluation metrics as **rewards** to guide a **neural reinforcement learning** based summarizer to generate summaries. A **genetic algorithm** based summarizer is also provided, which uses the unsupervised metrics as its **fitness function**.


Contact person: Yang Gao, yang.gao@rhul.ac.uk

https://sites.google.com/site/yanggaoalex/home

Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions

## Example Use Cases

### Evaluate Summaries (*evaluate_summary.py*)
Given the source documents and some to-be-evaluated summaries, you can produce the unsupervised metrics for the summaries with a few lines of code:

```python
from ref_free_metrics.sbert_score_metrics import get_sbert_score_metrics
from utils.data_reader import CorpusReader

reader = CorpusReader('data/topic_1')
source_docs = reader()
summaries = reader.readSummaries() 
scores = get_sbert_score_metrics(source_docs, summaries, pseudo_ref='top15')
```
In the example above, it extracts the top-15 sentences from each source document
to build the *pseudo reference summaries*, and rate the summaries
by measuring their semantic similarity with the pseudo references.

### Generate Summaries (*generate_summary_rl.py*) 
You can also use the unsupervised metrics as rewards to train a RL-based summarizer to generate summaries:

```python
# read source documents
reader = CorpusReader()
source_docs = reader('data/topic_1')

# generate summaries, with summary max length 100 tokens
rl_summarizer = RLSummarizer()
summary = rl_summarizer.summarize(source_docs, summ_max_len=100)

# print out the generated summary
print(summary)
```
You can also use the unsupervised metrics as the *fitness function* to guide a genetic algorithm to search for the optimal summary. See the example provided in *generate_summary_ga.py*.

If human-written reference summaries are available (assume they are at *data/topic_1/references*), you can also evaluate the quality of the generated summary against the references using **ROUGE**:

```python
refs = reader.readReferences() 
for ref in refs:
    rouge_scores = evaluate_summary_rouge(summary, ref)
```

## How to Set Up 
* Prerequisite: Python 3.6 or higher versions
* Install all packages in requirement.txt.
```shell script
pip3 install -r requirements.txt
```
* (Optional, if you want to run ROUGE) Download ROUGE-RELEASE-1.5.5.zip from the [link](https://drive.google.com/file/d/1eq4WD1rsCzAFhKmgI8cSeGqHEYYIFhGJ/view?usp=sharing), unzip the file and place extracted folder under the rouge directory
```shell script
mv ROUGE-RELEASE-1.5.5 rouge/
```

## License
Apache License Version 2.0
