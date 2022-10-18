# Text2Model 

Official implementation of ["_Text2Model: Model Induction for Zero-shot Generalization Using Task Descriptions_"](https://arxiv.org/).

We study the problem of generating a training-free task-dependent visual classifier from text descriptions without visual samples. 
We analyze the symmetries of T2M, and characterize the equivariance and invariance properties of corresponding models. In light of these properties we design an architecture based on hypernetworks that given a set of new class descriptions predicts the weights for an object recognition model which classifies images from those zero-shot classes. 
We demonstrate the benefits of our approach compared to zero-shot learning from text descriptions in image and point-cloud classification using various types of text descriptions: From single words to rich text descriptions.

### The text-to-model learning problem and our architecture
<p align="center"> 
    <img src="figs/Fig1.png" width="650">
</p>

### T2M-HN architecture
<p align="center"> 
    <img src="figs/Fig2.png" width="650">
</p>


### Main results
<p align="center"> 
    <img src="figs/Tab1.png" width="650">
</p>

<p align="center"> 
    <img src="figs/Tab2.png" width="650">
</p>

<p align="center"> 
    <img src="figs/Tab3.png" width="650">
</p>



## Installation 
- RUN: ```docker pull amosy3/t2m:latest```
- RUN: ```docker run -v <project_dir>:/data:rw --name <container_name> amosy3/t2m:latest```

## Get code and data
- Git clone this project to your <project_dir>
- Download and extract: 
[awa](https://chechiklab.biu.ac.il/~amosy/awa.zip),
[cub](https://chechiklab.biu.ac.il/~amosy/cub.zip)
[sun](https://chechiklab.biu.ac.il/~amosy/sun.zip) 
[descriptors](https://chechiklab.biu.ac.il/~amosy/gpt_label2descriptors.pkl)
[attributes](https://chechiklab.biu.ac.il/~amosy/label2attributes_names.pkl) 


## Run an experiment
- Use ```wandb login``` to login to you wandb account
- Run: ```python main.py --batch_size=64 --hn_train_epochs=100 --hnet_hidden_size=120 --inner_train_epochs=3 --lr=0.005 --momentum=0.9 --weight_decay=0.0001 --text_encoder SBERT --hn_type EV```




