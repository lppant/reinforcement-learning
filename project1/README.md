# Project 1 CS 7642 Spring 2020

#### Description

This project replicates Figure3, 4, 5 from Sutton 1988 paper and uses the following libraries:

* `python 2.7`
* `numpy`
* `matplotlib`

#### Installations
1. Ensure python 2.7 is installed on the machine running the code.
2. If numpy or matplotlib are not installed yet on the machine, then perform below installations: 
* `pip install numpy`
* `pip install matplotlib` 

#### How to run the project code?
To run project1 code, clone the repository, cd into `project1/code` subdirectory, and run `python sutton_experiments.py`.

This script will run the code necessary to plot replicas of Figures 3, 4, and 5 from Sutton's paper.
The run will create a new directory called `out` inside `project1` if it does not exist. 
For each run of the script, the below plot files will be generated inside a new directory within `out`:
* `figure_3.png`
* `figure_4.png`
* `figure_5.png`

Please note that Figure 5 relies on the results from Figure 4 computation.

Additionally, hyperparameters can be adjusted in `project1/code/hyperparameters.py`.
