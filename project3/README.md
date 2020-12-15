# Project 3 CS 7642 Spring 2020

#### Description

This project aims to replicate the Soccer Game experiments as illustrated in paper titled “Correlated-Q Learning” by Amy Greenwald and Keith Hall.
It requires the following key libraries (other dependencies are listed in `Installations` section below):

* `python 3.7`
* `numpy`
* `matplotlib`
* `cvsopt`

#### Installations
1. Ensure python 3.7 is installed on the machine running the code.
2. Ensure conda (any version) is installed on the machine.
2. Create conda environment from the given environment file.
* `conda env create -f environment.yml`
* `conda activate project3`

#### How to run the project code?
To run project3 code, clone the repository, cd into `project3/code` subdirectory, and run as below:
 * run `python main.py`.
 It creates 4 graphs: `Q-learning.png`, `Friend-Q.png`, `Foe-Q.png` and `Correlated-Q.png`.
 
Running these scripts will create a new directory called `out` inside `project3` if it does not exist. 
All graphs will be created within this `out` directory in a separate folder.