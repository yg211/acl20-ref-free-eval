## Using Supert for Single-Document Summarization: A Case Study on SummEval

Supert is originally designed to rate multi-document summaries, but it can also be used 
to rate single-document summaries. In this directory we
use Supert to evaluate summaries from the [SummEval](https://github.com/Yale-LILY/SummEval) 
dataset. The summaries
are generated using documents from CNN/DailyMail, and are rated by human experts
on their quality in *consistency*, *fluency*, *coherence* and *relevance*.


### Step 1: Prepare SummEval Data
Follow the instructions at [here](https://github.com/Yale-LILY/SummEval#data-preparation)
to prepare the data.
It will generate mulitple folders, but we only need *pairs/model_annotations.aligned.paired.jsonl*.
Move that jsonl file to this directory. It includes 100 topics, each has 23 summaries
and their human ratings.


### Step 2: Generate Supert scores
Simply run the following command:

```bash
python supert_summ_eval.py
```
It will use Supert to rate all summaries in the SummEval dataset and compute
the correlation between the Supert scores and the human ratings.
The default setup is to use the first 10 sentences from each document to build
the pseudo reference, but you could change the strategy at line 101 in *supert_summ_eval.py*:

```python
supert_scores = generate_supert_scores(summ_eval_data, 'top10') 
```
The generated Supert scores will also be saved at a pickle file to facilitate reuse.


### Results

The Pearson correlation between Supert scores and human ratings are below.

|               | Coherence | Fluency | Relavancy | Consistency |
|---------------|-----------|---------|-----------|-------------|
| Supert, top10 | .220      | .300    | .309      | .391        |
| Supert, top200| .221      | .287    | .311      | .389        |
