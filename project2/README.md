# Project 2 CS 7642 Spring 2020

#### Description

This project aims to perform experiments to successfully land “Lunar Lander” in the Open AI Gym environment.
It requires the following key libraries (other dependencies are listed in `Installations` section below):

* `python 3.6`
* `numpy`
* `matplotlib`
* `gym`
* `keras`
* `tensorflow`

#### Installations
1. Ensure python 3.6 is installed on the machine running the code.
2. Ensure conda (any version) is installed on the machine.
2. Create conda environment from the given environment file.
* `conda env create -f environment.yml`
* `conda activate project2` 

#### How to run the project code?
To run project2 code, clone the repository, cd into `project2/code` subdirectory, and run as below:
 * For training and testing with selected hyper-parameters and generating trained agent graphs, run `python lunar_lander.py`.
 It creates `train_scores.png` and `test_scores.png`.
 * For learning rate tuning and and generating related graph, run `python lunar_lander_lr.py`.
 It creates `learning_rate_scores.png`.
 * For gamma tuning and and generating related graph, run `python lunar_lander_gamma.py`.
 It creates `gamma_scores.png`
 * For epsilon decay tuning and and generating related graph, run `python lunar_lander_epsilon_decay.py`.
 It creates `epsilon_decay_scores.png`.
 
Running these scripts will create a new directory called `out` inside `project2` if it does not exist. 
All graphs will be created within this `out` directory in a separate folder.