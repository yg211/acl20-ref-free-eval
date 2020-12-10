# SUPERT: Unsupervised Multi-Document Summarization Evaluation & Generation

This project includes the source code for the paper [**SUPERT: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization**](https://www.aclweb.org/anthology/2020.acl-main.124.pdf), appearing at ACL 2020.

**Highlighted Features**

* *Unsupervised evaluation metrics*: Measure multi-document summaries without using human-written reference summaries
* *Unsupervised multi-document summarizer*: Using the unsupervised evaluation metrics as **rewards** to guide a **neural reinforcement learning** based summarizer to generate summaries. A **genetic algorithm** based summarizer is also provided, which uses the unsupervised metrics as its **fitness function**.


Contact person: Yang Gao, yang.gao@rhul.ac.uk

https://sites.google.com/site/yanggaoalex/home

Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions

## Example Use Cases

### Evaluate Multi-Document Summaries (*evaluate_summary.py*)
Given the source documents and some to-be-evaluated summaries, you can produce the unsupervised metrics for the summaries with the code below:

```python
from ref_free_metrics.supert import Supert
from utils.data_reader import CorpusReader

# read docs and summaries
reader = CorpusReader('data/topic_1')
source_docs = reader()
summaries = reader.readSummaries() 

# compute the Supert scores
supert = Supert(source_docs, ref_metric='top15') 
scores = supert(summaries)
```
In the example above, it extracts the top-15 sentences from each source document
to build the *pseudo reference summaries*, and rate the summaries
by measuring their semantic similarity with the pseudo references.

### Evaluate Single-Document Summaries 
You could use the same code for evaluating multi-doc summaries to rate single-doc summaries.
In addition to that, you may consider using more sentences from the input doc to 
build the pseudo reference, by replacing argument 'top15' in the above code by, e.g., 'top30',
so as to use the first 30 (instead of 15) sentences to build the pseudo reference.

We study the influence of pseudo reference length on the performance of Supert 
for single-doc summaries at [summ_eval](summ_eval/). We compare
correlation between Supert and human ratings from the [SummEval](https://github.com/Yale-LILY/SummEval)
dataset.


### Generate Summaries (*generate_summary_rl.py*) 
You can also use the unsupervised metrics as *rewards* to train a RL-based summarizer to generate summaries:

```python
from generate_summary_rl import RLSummarizer

# read source documents
reader = CorpusReader('data/topic_1')
source_docs = reader()

# generate summaries using reinforcement learning, with supert as reward function
supert = Supert(source_docs)
rl_summarizer = RLSummarizer(reward_func = supert)
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

## Reproduce the results in the paper
* Branch [**compare\_metrics**](https://github.com/yg211/acl20-ref-free-eval/tree/compare_metrics) provides the code for reproducing the results in Tables 1 - 4. 
* Branch [**tac\_summarisation**](https://github.com/yg211/acl20-ref-free-eval/tree/tac_summarisation) provides the code for reproducing the results in Table 5.

## License
Apache License Version 2.0
