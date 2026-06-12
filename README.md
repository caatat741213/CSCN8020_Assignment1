## **Reinforcement Learning Programming-Assignment 1**

#### **Student name:** Chao-Chung, Liu
#### **Student ID:** 9067679
#### **Link:** https://github.com/caatat741213/CSCN8020_Assignment1.git

## Short summary of the assignment

The assignment focuses on applying Dynamic Programming and Monte Carlo methods to solve Markov Decision Processes (MDPs). It includes manual mathematical derivations for a 2x2 Gridworld, implementation of Value Iteration (standard and in-place) for a 5x5 Gridworld, and Off-policy Monte Carlo with Importance Sampling.


## Description of the repository structure
```
CSCN8020_Assignment1/
├── .gitignore          # excludes virtual environments, cache files, checkpoints, and generated runtime files.
├── README.md
├── requirements.txt    # dependencies needed to run the notebook
├── CSCN8020_Assignment1.pdf
├── CSCN8020_Assignment1.ipynb  # single notebook containing all four problems
├── images/             # figures and screenshots for the report
├── logs/               # saved experiment or evaluation logs
└── src/
    ├── __init__.py
    ├── agents.py
    ├── environments.py
    ├── policies.py
    └── utils.py
```

## Any assumptions or known limitations
1. A discount factor (γ) of 0.9 is assumed for the Gridworld calculations
2. The transitions for the pick-and-place robotic arm in Problem 1 are assumed to be deterministic for simplicity.
3. The Off-Policy Monte Carlo algorithm uses a completely random behavior policy (25% probability for each action). A known limitation of this approach is high variance, which requires a significantly larger number of episodes to properly converge compared to Value Iteration.


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
* **Textbook:** Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
* **Development Library:** [Gymnasium Documentation](https://gymnasium.farama.org/) - Used as the standard API for environment creation.
* **Course Code References :**
  * MDP Basics: [lec2_MDP](https://github.com/CSCN8020/playground/tree/main/lec2_MDP)
  * Dynamic Programming: [lec3_DP](https://github.com/CSCN8020/playground/tree/main/lec3_DP)
  * Monte Carlo: [lec4_MC](https://github.com/CSCN8020/playground/tree/main/lec4_MC)
  * GymMaze Example: [HelloGymMaze](https://github.com/ProfEspinosaAIML/HelloGymMaze.git)
