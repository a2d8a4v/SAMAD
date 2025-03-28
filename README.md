# [SAMAD](https://ieeexplore.ieee.org/document/10832269) (Speech Assessment Multi-Aspect Design)
<img src="/icon/SAMAD.png" alt="" width="578" height="450">

## Package Requirements

Install the following packages in your virtual environment before running the code:

- python version 3.8.16

You can run the following command in your virtual environment:

```pip install -r requirements.txt```
	

## Training and Evaluation

Run
```bash train.sh```
```bash test_submodule.sh```

Note that every run does not produce the same results due to the random elements.


## What is SAMAD?
SAMAD is a speech assessment model which consists of three important ingredients: **prompt-relevance, delivery, and language use.** Each module is designed to specific criteria essential for the evaluation of spontaneous spoken responses.

## Features
* **Optimized with soft labels to emphasize the ordinal properties of automatic speech assessment tasks.**
* **Combines self-supervised pre-trained (SSL) models with handcrafted indicator features to enrich the representation of spoken responses.**
* **Specifically designed to target criteria essential for the evaluation of spontaneous spoken responses.**

## How to get the dataset?
* To obtain the dataset, please reach out to the Language Training & Testing Center (LTTC).

## Results
Enhanced existing models, successfully improving recognition accuracy by 9% in a 5-scale classification task

<img src="/icon/seen.png" alt="" width="574" height="182">


## Citation
If you find this work helpful, please consider citing us:
```bibtex
@INPROCEEDINGS{10832269,
  author={Peng, Wen-Hsuan and Chen, Sally and Chen, Berlin},
  booktitle={2024 IEEE Spoken Language Technology Workshop (SLT)}, 
  title={Enhancing Automatic Speech Assessment Leveraging Heterogeneous Features and Soft Labels For Ordinal Classification}, 
  year={2024},
  pages={945-952},
  keywords={Adaptation models;Conferences;Neural networks;Self-supervised learning;Speech enhancement;Optimization;Automated speech assessment;Multi-modal model;End-to-end neural network},
  doi={10.1109/SLT61566.2024.10832269}}

