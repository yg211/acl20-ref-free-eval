# SUPERT: Unsupervised Summarization Evaluation & Generation

This project includes the source code for ACL-2020 paper xxx. 

**Highlighted Features**

* *Unsupervised evaluation metrics*: Measure multi-document summaries without using human-written reference summaries
* *Unsupervised multi-document summarizer*: Using the unsupervised evaluation metrics as **rewards** to guide a **neural reinforcement learning** based summarizer to generate summaries.

## Example Use Cases

### Evaluate Summaries

### Generate Summaries
*generate_summary.py* provides an example for using the unsupervised metric as rewarads
to train a RL-based summarizer. 
```python
# read source documents
reader = CorpusReader(BASE_DIR)
source_docs = reader('data/topic_1/input_docs')

# generate summaries, with summary max length 100 tokens
rl_summarizer = RLSummarizer()
summary = rl_summarizer.summarize(source_docs, summ_max_len=100)

# print out the generated summary
print(summary)
```
In this example, it is assumed that the to-be-summarized documents are in the directory *data/topic_1/input_docs*, and the generated summary should contain no more than 100 tokens.

## Prerequisites
* Python 3.6 or higher versions
* Install all packages in requirement.txt.
```shell script
pip3 install -r requirements.txt
```

