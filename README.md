# SUPERT: Unsupervised Summarization Evaluation & Generation

This project includes the source code for ACL-2020 paper xxx. 

**Highlighted Features**

* *Unsupervised evaluation metrics*: Measure multi-document summaries without using human-written reference summaries
* *Unsupervised multi-document summarizer*: Using the unsupervised evaluation metrics as **rewards** to guide a **neural reinforcement learning** based summarizer to generate summaries.

## Example Use Cases

### Evaluate Summaries
Given the source documents and some to-be-evaluated summaries, you can produce the unsupervised metrics for the summaries with a few line of codes:

```python
from ref_free_metrics.sbert_score_metrics import get_sbert_score_metrics
from utils.data_reader import CorpusReader

reader = CorpusReader('data/topic_1')
source_docs = reader()
summaries = reader.readSummaries() 
scores = get_sbert_score_metrics(source_docs, summaries, pseudo_ref='top15')
# output scores: [0.47249420469344194, 0.3846229787455767, 0.4947284578016637, 0.5013696306058708, 0.5108337611651548] 
```
In the example above, it extracts the top-15 sentences from each source document
to build the *pseudo reference summaries*, and rate the summaries
by measuring their semantic similarity with the pseudo references.
*generate_summary.py* provides a working example.

### Generate Summaries
*generate_summary.py* provides an example for using the unsupervised metric as rewarads
to train a RL-based summarizer. 
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

If human-written reference summaries are available (assume they are at *data/topic_1/references*), you can also evaluate the quality of the generated summary against the references using ROUGE:

```python
refs = reader.readReferences() 
for ref in refs:
    rouge_scores = evaluate_summary_rouge(summary, ref)
```

With the provided sample data (located at *data/topic_1*), the generated summary is 

```
A man intent on committing suicide left his car on a railroad track in Glendale near downtown Los Angeles Wednesday, where it set off a collision that derailed two commuter trains, killing at least 10 people and injuring nearly 200, authorities said. One of the stricken trains then side-swiped a stationary freight train that was parked on a side-track in the area, knocking it over, officials said. Authorities believe Juan Manuel Alvarez drove his Jeep Grand Cherokee around the rail crossing barrier and onto the Metrolink train tracks at Chevy Chase Drive in Glendale about 6 a.m. Wednesday
```
And its ROUGE scores against the golden references are:
```
ROUGE-1:	0.42
ROUGE-2:	0.09091
ROUGE-L:	0.22
ROUGE-SU4:	0.15251
```
Please note that, due the to stochastic nature of RL, the generated summaries at different runs 
are most likely to be different.



## Prerequisites
* Python 3.6 or higher versions
* Install all packages in requirement.txt.
```shell script
pip3 install -r requirements.txt
```

