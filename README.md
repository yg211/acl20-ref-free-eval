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
reader = CorpusReader()
source_docs = reader('data/topic_1/input_docs')

# generate summaries, with summary max length 100 tokens
rl_summarizer = RLSummarizer()
summary = rl_summarizer.summarize(source_docs, summ_max_len=100)

# print out the generated summary
print(summary)
```

In this example, it is assumed that the to-be-summarized documents are in the directory *data/topic_1/input_docs*, and the generated summary should contain no more than 100 tokens. If human-written reference summaries are available (assume they are at *data/topic_1/references*), you can also evaluate the quality of the generated summary against the references using ROUGE:

```python
refs = reader.readReferences() # make sure you have put the references in data/topic_1/references
avg_rouge_score = {}
for ref in refs:
    rouge_scores = evaluate_summary_rouge(summary, ref)
    add_result(avg_rouge_score, rouge_scores)
print('\n=====Average ROUGE scores against {} references====='.format(len(refs)))
for metric in avg_rouge_score:
    print('{}:\t{}'.format(metric, np.mean(rouge_scores[metric])))
```

With the provided sample data (located at *data/topic_1*), the generated summary is 

```
A ``deranged'' man abandoned his SUV on railroad tracks when he aborted a suicide attempt, authorities said, and then watched as a Metrolink commuter train slammed into it, setting off a spectacular three-train collision that left at 11 least dead and more than 180 injured Wednesday. After derailing, the commuter train, which was being pushed by its engine, struck a Union Pacific locomotive parked on a side track, knocking it onto its side, fire department officials said. A 25-year-old man was arrested following Wednesday's commuter train collision that killed at least 10 people near downtown Los Angeles.
```
And its ROUGE scores against the golden references are:
```
ROUGE-1:	0.39
ROUGE-2:	0.0303
ROUGE-L:	0.19
ROUGE-SU4:	0.11428
```
Please note that, due the to stochastic nature of RL, the generated summaries at different runs 
are most likely to be different.


## Prerequisites
* Python 3.6 or higher versions
* Install all packages in requirement.txt.
```shell script
pip3 install -r requirements.txt
```

