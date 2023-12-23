class Hypergraph:

  def __init__(self, H0 = {'0': [0, 1]}):

    # initialize graph as initial hypergraph H0
    self.__H = H0

    #  number of hyperedges
    self.__m = len(self.__H)

    # number of nodes
    self.__n = len(set([v for f in self.__H.values() for v in f]))

  def empirical_node_pmf(self, verbose=False):

    '''
    Form an empirical probability mass function of node degrees
    '''

    # count number of hyperedges that each vertex is in
    counter = Counter(v for e in self.__H.values() for v in e)

    # compute empirical probability distribution
    vertices, degree_sum = list(counter.keys()), sum(counter.values())
    probabilities = [deg / degree_sum for deg in counter.values()]

    if verbose:
      print(dict(zip(vertices, probabilities)))

    return vertices, probabilities

  def display(self, save=False):

    '''
    Display hypergraphs using HyperNetX for sufficiently small structures
    '''

    output = hnx.Hypergraph(self.__H)
    hnx.drawing.draw(output, with_edge_labels=True, with_node_labels=True)
    if save:
        plt.savefig("hypergraph.png", dpi=300, bbox_inches='tight')


  def sample_nodes(self):

    '''
    According to the empirical pmf, select a node with probability proportion to node degree
    '''

    nodes, probabilities = self.empirical_node_pmf(verbose=False)
    node = np.random.choice(nodes, size=None, p=probabilities)

    return node

  def empirical_hyperedge_pmf(self, verbose=False):

    '''
    Form an empirical probability mass function of hyperedge cardinalities
    '''

    # count number of vertices in each hyperedge
    card_dict = {key: len(value) for key, value in self.__H.items()}

    # compute empirical probability distribution
    hyperedges, cardinality_sum = list(card_dict.keys()), sum(card_dict.values())
    probabilities = [card / cardinality_sum for card in card_dict.values()]

    if verbose:
        print(dict(zip(hyperedges, probabilities)))

    return hyperedges, probabilities

  def sample_hyperedges(self):

      '''
      According to the empirical pmf, select a hyperedge with probability proportion to its cardinality
      '''

      hyperedges, probabilities = self.empirical_hyperedge_pmf(verbose=False)
      hyperedge = np.random.choice(hyperedges, size=None, p=probabilities)

      return str(hyperedge)


  def grow(self, N, k=1):

    '''
    Run the M-model from Roh et al. for a specified number of timesteps (N)
    At each timestep, perform a hyperedge-addition event by node-based preferential attachment
    and a hyperedge-addition event by hyperedge-based preferential attachment
    For the hyperedge-addition event, add k vertices (default: k=1)

    We consider these events to occur concurrently, so the size (k+1)-hyperedge added
    in the node-based preferential attachment step does not appear
    '''

    for i in tqdm(range(0, N)):

        # Node-based preferential attachment step
        # Added size (k+1)-hyperedge
        self.__n += k
        hyperedge_new_temp = [self.sample_nodes()] + [self.__n - k + i for i in range(k)]

        # Hyperedge-based preferential attachment step
        chosen_hyperedge = self.sample_hyperedges()
        # Form expanded hyperedge
        self.__H[chosen_hyperedge] = self.__H[chosen_hyperedge] + [self.__n - k + i for i in range(k)]
        # Add node-based (k+1)-cardinality hyperedge
        self.__H[f"{self.__m}"] = hyperedge_new_temp
        # Iterate number of hyperedges
        self.__m += 1

  def get_degree(self, vertex):

    '''
    Return the degree of a labeled vertex
    '''

    return len({k: v for k, v in self.__H.items() if vertex in v})

  def n_nodes(self):

      return self.__n

  def m_hyperedges(self):

      return self.__m

  def get_H(self):

      return self.__H

  def card_sum(self):

    return sum([len(he) for he in self.get_H().values()])

  def node_deg_sum(self):

    return sum([len({k: v for k, v in self.get_H().items() if vertex in v}) for vertex in range(self.__n)])

  def get_incidence_matrix(self):

    incidence_matrix = np.zeros((self.__n, self.__m))

    for hyperedge, nodes in hg1.get_H().items():
        for node in nodes:
            incidence_matrix[int(node), int(hyperedge)] = 1

    return incidence_matrix

  def get_degree_matrix(self):

      degree_matrix = np.zeros((self.__n, self.__n))

      for i in range(self.__n):
          degree_matrix[i,i] = hg1.get_degree(i)

      return degree_matrix

  def get_adjacency_matrix(self):

    return (self.get_incidence_matrix() @ self.get_incidence_matrix().T - self.get_degree_matrix())

  def pmf_to_cdf(self, di):

    '''
    Using dictionary corresponding to probability mass function,
    generated a cumulative degree distribution function
    '''

    sorted_di = dict(sorted(di.items(), reverse=True))
    freq_list = list(sorted_di.values())
    csum_li = np.cumsum(freq_list)
    cdf_dict = dict(zip(sorted_di.keys(), csum_li / csum_li[-1]))

    return cdf_dict


  def deg_dist_cdf(self, name="", to_csv=False):

    '''
    Form an empirical cumulative distribution function (cdf) for node degrees
    to_csv allows exportation to a CSV file
    '''

    counts = dict()
    for hyperedge in self.__H.values():
        for vertex in hyperedge:
            if vertex in counts:
                counts[vertex] += 1
            else:
                counts[vertex] = 1
    inv_d = defaultdict(int)

    for k, v in counts.items():
        inv_d[v] += 1
    inv_d = dict(inv_d)
    inv_d = {k: v / sum(inv_d.values()) for k, v in inv_d.items()}
    inv_d = dict(sorted(inv_d.items(), reverse=True))

    cdf = self.pmf_to_cdf(inv_d)

    if to_csv:
        df = pd.DataFrame({"degree" : list(cdf.keys()), "cum_prob" : list(cdf.values())})
        df.to_csv(f"degree_data_{name}.csv", index=False)

    return cdf

  def card_cdf(self, to_csv=False):

    '''
    Form an empirical cumulative distribution function for hyperedge cardinality
    '''

    d = dict()
    cardinality_list = [len(he) for he in self.__H.values()]

    for card in cardinality_list:
        if card in d.keys():
            d[card] += 1
        else:
            d[card] = 1

    d = {k: v / sum(d.values()) for k, v in d.items()}
    d = dict(sorted(d.items(), reverse=True))

    cdf = self.pmf_to_cdf(d)

    if to_csv:
        df = pd.DataFrame({"cardinality" : list(cdf.keys()), "cum_prob" : list(cdf.values())})
        df.to_csv("cardinality_data.csv", index=False)

    return cdf