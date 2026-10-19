# Landmark-pruned reverse-nearest-neighbour queries on road networks

Reverse nearest neighbour (RNN) moving queries ask which data objects would have a moving query point as their nearest neighbour. This task is central to location-based services, and the adjustable-tree paper "An adjustable-tree method for processing reverse nearest neighbor moving queries" (2019) proposes an index for it on road networks. This experiment tests a simpler idea from the same problem family: precomputed landmark distance bounds can prune candidate objects before any exact shortest-path check.

**Question:** Does a landmark-pruned index answer reverse-nearest-neighbour queries on a road network faster than a standard exhaustive baseline, at the same exact answers, as the density of data objects grows?

**Dataset:** A synthetic perturbed grid road network, generated in networkx with random edge removal and random edge weights. Data objects are sampled at several densities, and moving queries are short random walks on the graph. This is not the paper's dataset, and no public road network is downloaded.

**Baseline:** An exhaustive exact RNN check. For each query, it runs a full Dijkstra from every data object and tests whether the query is that object's nearest neighbour. It has no index.

**Improved model:** The same exact check, with a landmark index added. Triangle-inequality lower and upper bounds prune objects first, and the same Dijkstra verification runs only on the surviving candidates. Both methods must return identical answers.

**How to run:**
```
pip install -r requirements.txt
python analysis.py
```

**Results:** Results are written to `results.json`, and a latency chart is saved to `figures/`.

## Reference

This project is motivated by the following paper:

> YI Chang, JH Shen, CM Chu (2019). *An adjustable-tree method for processing reverse nearest neighbor moving queries*. International Conference on Frontier. https://link.springer.com/chapter/10.1007/978-981-15-3250-4_43

Paper author: Prof. Ye-In Chang, National Sun Yat-sen University (changyi@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
