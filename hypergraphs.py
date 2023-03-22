!pip install hypernetx

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import hypernetx as hnx
from scipy.stats import bernoulli
from collections import Counter, defaultdict
import random
import seaborn as sns

class Hypergraph:
  def __init__(self, H0 = {'0': [0]}, alpha = 0.5, p = 0.5):
    self.p = p
    # initialize growing hypergraph H as initial hypergraph H0
    self.H = H0
    # get hypergraph properties
    self.n_hyperedges = len(list(set([value for values in self.H.values() for value in values])))
    # get number of vertices in hypergraph
    self.n_vertices = len(self.H.keys())
    # parameter to control upper bound for vertex sampling at each step
    self.alpha = alpha

  def empirical_distribution(self, verbose=False):
    # form an empirical probability distribution over vertex degrees 
    # for preferential attachment mechanism
    counter = Counter([value for values in self.H.values() for value in values])
    vertex_set = list(counter.keys())
    total = sum(counter.values())
    probabilities = [val / total for val in counter.values()]
    if verbose:
      print(dict(zip(vertex_set, probabilities)))
    return vertex_set, probabilities

  def select_vertices(self, upper_bound, add_vertex=False, uniform=False):
    # according to empirical distribution, sample vertices with replacement
    vertex_set, probabilities = self.empirical_distribution(verbose = False)
    if uniform is False:
        if add_vertex is False:
          vertices = list(np.random.choice(vertex_set, 
                                                     size=self.sample_hyperedge_cardinality(upper_bound), 
                                                     p=probabilities, replace=True))
        elif add_vertex is True:
          vertices = list(list(np.random.choice(vertex_set, 
                                                     size=self.sample_hyperedge_cardinality(upper_bound) - 1, 
                                                     p=probabilities, replace=True)) + [self.n_vertices])
    else:
        if add_vertex is False:
          vertices = list(np.random.choice(vertex_set, 
                                                 size=upper_bound, 
                                                 p=probabilities, replace=True))
        elif add_vertex is True:
          vertices = list(list(np.random.choice(vertex_set, 
                                                 size=upper_bound-1, 
                                                 p=probabilities, replace=True)) + [self.n_vertices])
    return vertices
      
  def sample_hyperedge_cardinality(self, i):
    # determine how many vertices in new hyperedge
    # sampled uniformly at randomly (otherwise 2)
    upper_bound = math.floor(pow(i, self.alpha))
    # if upper bound is sufficiently greater than or equal to 2
    if upper_bound >= 2:
        return random.randint(2, upper_bound)
    # if not large enough
    return 2

  def sample_bernoulli(self):
    # sample from a Bernoulli random variable to determine if timestep will add a vertex
    b = bernoulli.rvs(self.p, size=1)[0]
    return b

  def display(self):
    # use HyperNetX to display current hypergraph
    output = hnx.Hypergraph(self.H)
    # draw hypergraph, suppress edge and node labels
    hnx.drawing.draw(output, 
                     with_edge_labels=False,
                     with_node_labels=True)

  def modified_avin(self, t):
    # run modified Avin et al. model for specified number of timesteps (t)
    for i in range(0, t):
      # sample from Bernoulli random variable
      b = self.sample_bernoulli()
      if (b == 1):
        # create new hyperedge with y - 1 vertices and new vertex
        self.H[str(self.n_hyperedges)] = self.select_vertices(i, add_vertex = True)
        self.n_vertices += 1
        self.n_hyperedges += 1
      else:
        # create new hyperedge with y vertices
        self.H[str(self.n_hyperedges)] = self.select_vertices(i)
        self.n_hyperedges += 1
        
  def d_uniform(self, k, t):
    # generate k-uniform hypergraphs for some choice of hyperedge cardinality (d) and timesteps (t)
    # for later implementation
    for i in range(0, t):
        b = self.sample_bernoulli()
        if (b == 1):
            # create new hyperedge with k - 1 vertices and new vertex
            self.H[str(self.n_hyperedges)] = self.select_vertices(k, add_vertex = True, uniform = True)
            self.n_vertices += 1
            self.n_hyperedges += 1
        else:
            # create new hyperedge with k vertices
            self.H[str(self.n_hyperedges)] = self.select_vertices(k, uniform = True)
            self.n_hyperedges += 1