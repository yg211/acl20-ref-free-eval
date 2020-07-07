# SUPERT: Unsupervised Multi-Document Summarization Evaluation & Generation

This branch provides the code for running the Supert-based summarisers on TAC08 and 09 datasets.

## How to Set Up 
* Prerequisite: Python 3.6 or higher versions
* Download TAC08 from [here](https://tac.nist.gov/data/past/2008/UpdateSumm08.html) and TAC09 from [here](https://tac.nist.gov/data/past/2009/Summ09.html), and put the downloaded data to the directory *data/*.
* Install all packages in *requirement.txt*.
```shell script
pip3 install -r requirements.txt
```
* (Optional, if you want to run ROUGE) Download ROUGE-RELEASE-1.5.5.zip from the [link](https://drive.google.com/file/d/1eq4WD1rsCzAFhKmgI8cSeGqHEYYIFhGJ/view?usp=sharing), unzip the file and place extracted folder under the rouge directory
```shell script
mv ROUGE-RELEASE-1.5.5 rouge/
```

## How to run
Two summarisers are provided, one based on reinforcement learning (RL) based and the other based on genetic algorithm. For example, you can run the RL-based summariser on TAC08 by using the command below.
```shell script
python generate_summary_rl.py 08
```

## License
Apache License Version 2.0
