## **Reinforcement Learning Programming-Assignment 1**

#### **Student name:** Chao-Chung, Liu
#### **Student ID:** 9067679
#### **Link:** https://github.com/caatat741213/CSCN8020_Assignment1.git

## Short summary of the assignment

The assignment focuses on applying Dynamic Programming and Monte Carlo methods to solve Markov Decision Processes (MDPs). It includes manual mathematical derivations for a 2x2 Gridworld, implementation of Value Iteration (standard and in-place) for a 5x5 Gridworld, and Off-policy Monte Carlo with Importance Sampling.


## Description of the repository structure
```
CSCN8020_Assignment1/
├── .gitignore          #excludes virtual environments, cache files, checkpoints, and generated runtime files.    
├── requirements.txt    #dependencies needed to run the notebook.    
├── README.md   
├── CSCN8020_Assignment1.pdf            
├── CSCN8020_Assignment1.ipynb  # Single notebook — all four problems
├── src/     
│   ├── __init__.py
│   ├── environments.py  
│   ├── agents.py      
│   ├── policies.py     
│   └── utils.py       
├── images/  
└── logs/              
```

## Any assumptions or known limitations
1. A discount factor (γ) of 0.9 is assumed for the Gridworld calculations



## Instructions to run the notebook

1. Clone the repository:

```bash
git clone https://github.com/caatat741213/CSCN8020_Assignment1.git
cd CSCN8020_Assignment1
```

2. Create and activate a virtual environment:

```bash
#Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Open Jupyter:

```bash
jupyter notebook CSCN8020_Assignment1.ipynb
```

5. Run the notebook from top to bottom.

## References
* Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.