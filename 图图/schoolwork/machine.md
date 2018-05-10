# Introduction
- Environment Types:  Fully observable --access to the complete state of the
environment at each point(vs. partially observable); Deterministic --next state of the environment is completely determined
by the current state and the action(vs. stochastic); Episodic --The choice of action in each episode depends only on the episode itself(vs. sequential)

- ![environment type](5.10.png)
# Search
- BFS vs DFS queue vs stack: stack may be more convenient to implement by recrusion but can overload; target is shallow - BFS otherwise - DFS;

- ![compare](conpare.png)
- A* tree search is optimal if it uses an admissible heuristic;
# CSP
- AC3: head <- tail, only update tail;
- Minimum remaining values (MRV)->Degree Heuristic (Deg:Choose the variable with the most constraints on remaining variables)->LCV(Choose the value that rules out the fewest values in the remaining variables)
- If a CSP has a solution, min-conflicts is not guaranteed to find it
# Adversarial Search
