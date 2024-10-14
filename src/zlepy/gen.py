import numpy as np
import itertools
from collections import defaultdict, deque
import sympy as sp

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def perm_inverse(perm):
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(inv), dtype=inv.dtype)
    return inv

#can speed-up by taking the fastest algorithm for finding disconnected components
def po_groups(po):
    groups = [po[0:2]]
    for i in range(1,int(len(po)/2)):
        relation = po[2*i:2*i+2]
        booli = 1
        #can map isin to avoid looping
        for j in range(len(groups)):
            if (np.any(np.in1d(relation,groups[j]))): #check if any node in relation belongs to given group j
                groups[j]=np.append(groups[j], tuple(relation)) #add to group if so, then break
                booli = 0
                break
        if (booli):
            groups.append(relation)
    return groups

def edges_to_adjacency_list(edges):
    adjacency_list = defaultdict(list)
    for i in range(0, len(edges), 2):
        u = edges[i]
        v = edges[i + 1]
        adjacency_list[u].append(v)

    # Ensure all nodes are included, even if they have no outgoing edges
    nodes = set(edges)
    for node in nodes:
        if node not in adjacency_list:
            adjacency_list[node] = []

    return adjacency_list

def add_missing_nodes(edge_list):
    all_nodes = set(edge_list.keys())

    # Include nodes that are only in the values
    for nodes in edge_list.values():
        all_nodes.update(nodes)

    # Ensure every node has a key in the dictionary
    for node in all_nodes:
        if node not in edge_list:
            edge_list[node] = []

    return edge_list

def topological_sort(graph):
    in_degree = defaultdict(int)
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    zero_in_degree_queue = deque([node for node in graph if in_degree[node] == 0])
    topo_order = []

    count = 0
    while zero_in_degree_queue:
        node = zero_in_degree_queue.popleft()
        topo_order.append(node)
        count += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_queue.append(neighbor)

    return topo_order

def longest_path_dag(graph):
    topo_order = topological_sort(graph)
    dist = {node: float('-inf') for node in graph}

    for node in graph:
        if node not in dist or dist[node] == float('-inf'):
            dist[node] = 0

    for node in topo_order:
        for neighbor in graph[node]:
            if dist[neighbor] < dist[node] + 1:
                dist[neighbor] = dist[node] + 1

    return dist

def transitive_reduction(graph):
    topo_order = topological_sort(graph)
    longest_paths = longest_path_dag(graph)
    reduced_graph = defaultdict(list)

    # collect longest paths
    for node in topo_order:
        for neighbor in graph[node]:
            if longest_paths[neighbor] == longest_paths[node] + 1:
                reduced_graph[node].append(neighbor)

    # ensure all nodes are preserved and maintain reachability
    for node in graph:
        if node not in reduced_graph:
            reduced_graph[node] = []
        for neighbor in graph[node]:
            if neighbor not in reduced_graph[node]:
                reduced_graph[node].append(neighbor)

    # remove shorter paths by checking if any node has more than one parent in the reduced graph
    for node in graph:
        for neighbor in graph[node]:
            for descendant in reduced_graph[neighbor]:
                if descendant in reduced_graph[node]:
                    reduced_graph[node].remove(descendant)

    return reduced_graph

#levels / by-variable permutation restriction
def level_sort(graph):
    in_degree = defaultdict(int)
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    zero_in_degree_queue = deque([node for node in graph if in_degree[node] == 0])
    zero_curr = [node for node in graph if in_degree[node] == 0]

    count = 0
    levels = defaultdict(list)
    levels[count] = zero_curr

    while zero_in_degree_queue:
        if (not any(np.isin(zero_curr, zero_in_degree_queue))):
            zero_curr = list(set([node for node in graph if in_degree[node] == 0])-set([x for xs in list(levels.values()) for x in xs]))
            count += 1
            levels[count] = zero_curr

        node = zero_in_degree_queue.popleft()
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_queue.append(neighbor)

    return levels

def reverse_graph(graph):
    reversed_graph = defaultdict(list)
    for source, targets in graph.items():
        for target in targets:
            reversed_graph[target].append(source)
    return reversed_graph

#count the restrictions from above and below on each node.
def get_above_below(nodes, g):
    rg=add_missing_nodes(reverse_graph(g))
    tpg = topological_sort(g)
    tpgr = topological_sort(rg)

    above_below={}
    for node in nodes:
        #forward graph
        R = [node]
        for i in tpg:
            if i in R:
                R = R + g[i]

        #reverse graph
        R2 = [node]
        for i in tpgr:
            if i in R2:
                R2 = R2 + rg[i]

        above_below[node] = [len(R2)-1, len(R)-1]

    return above_below

def lstlist2(lst):
    return [list([*i]) for i in lst]

def filter_duplicates(lists):
    return [lst for lst in lists if all(x != y for x, y in zip(lst, lst[1:]))]

def products2(lst):
    blocks = []
    for v in lst:
        blocks.append(lstlist2(filter_duplicates(list(itertools.product(*v)))))
    return blocks

#produce permutations for a given disconnected component
def get_perms(n, nodes, above_below, levels):
    legal_values = [nodes[0+i[0]:n-i[1]] for i in above_below.values()]
    #node_map deals with node sets that aren't 0-based and leaves them in the same order.
    node_map = dict(zip(nodes,np.arange(len(nodes))))
    block_variables=[[list(legal_values[node_map[j]]) for j in i] for i in list(levels.values())]
    ex2=products2(block_variables)
    res = [[elem for s in i for elem in s] for i in list(itertools.product(*ex2))] #these are products of all possibilities
    res = [i for i in res if len(set(i))==len(i)] #these are filtered plausible products
    level_order = [elem for s in list(levels.values()) for elem in s]
    #later convert to all np
    reorder = list(perm_inverse([node_map[i] for i in level_order]))
    res = [list(np.array(i)[reorder]) for i in res] #[[i[j] for j in level_order] for i in res]
    return res

def check(lis, order, nodesi):
    node_map = dict(zip(nodesi, np.arange(len(nodesi))))
    for i in range(0,int(len(order)/2)):
        rel = order[2*i:2*i+2]
        if (lis[node_map[rel[0]]] > lis[node_map[rel[1]]]):
            return False
    return True

#2 macro methods for user convenience
def fixed_perms(group_pos, group_nodes):
    filtered_graph = [add_missing_nodes(transitive_reduction(j)) for j in [edges_to_adjacency_list(i) for i in group_pos]]

    dc_perms=[]
    for dcompi in range(len(filtered_graph)):
        nodesi = group_nodes[dcompi]
        poi = group_pos[dcompi]
        n = len(nodesi)
        levels = level_sort(filtered_graph[dcompi])

        above_below = get_above_below(nodesi, filtered_graph[dcompi])#nodes_above_below(nodesi,levels, filtered_graph[dcompi])

        res = get_perms(n, nodesi, above_below, levels)
        #filter out impossible sub-lists
        checks=[check(i, poi, nodesi) for i in res]
        res = [list(i) for i in np.array(res)[checks]] #[list(np.array(i)[checks]) for i in res]

        dc_perms.append(res)

    #then product across
    canonical_perms = [[elem for s in i for elem in s] for i in list(itertools.product(*dc_perms))]
    return canonical_perms

def comb_labelings(nodes, group_nodes):
    #fencepost
    products =itertools.combinations(nodes, len(group_nodes[0]))
    products = [list(j) for j in products]

    for i in range(1,len(group_nodes)-1):
        subsets=itertools.combinations(nodes, len(group_nodes[i]))
        subsets = [list(j) for j in subsets]
        #sorting here unnecessary
        possible_labels=[[list(np.sort(list(set(j) - set(products[k])))) for j in subsets if len(list(set(j) - set(products[k]))) == len(group_nodes[i])] for k in range(len(products))]
        factors = [[[products[i]], possible_labels[i]] for i in range(len(products))]
        #need to flatten, can reduce these to one line
        products = [[[elem for s in i for elem in s] for i in list(itertools.product(*factors[j]))] for j in range(len(factors))]
        products = [elem for s in products for elem in s]

    #fencepost (saves time of taking combinations)
    remainder = [list(np.sort(list(set(nodes) - set(i)))) for i in products]
    if (remainder != [[]]):
        composite = [products[i]+remainder[i] for i in range(len(products))]
    else:
        composite = products

    return composite

def get_phi(nodes, group_pos, group_nodes, prints=False):
    #canonical perms is the list of permutations allowed for the fixed node labeling 0...n.
    canonical_perms = fixed_perms(group_pos, group_nodes)
    #composite is the list of possible node labelings (up to combination),
    combination_labels = comb_labelings(nodes, group_nodes)
    if (prints):
        print("fixed-order (1..n) permutations: ", len(canonical_perms), canonical_perms)#[0:3], "\n")
        print("node labels (up to combination): ", len(combination_labels), combination_labels)#[0:3])

    #permutation list phi is the canonical perms under all node labelings up to combination
    phi = []
    for i in combination_labels:
        dictt = dict(zip(nodes, i))
        conversions =[[dictt[j] for j in k] for k in canonical_perms]
        phi.append(conversions)
    phi = [np.array(j) for j in [elem for s in phi for elem in s]]
    return phi

def filtrate(perm):
    sd=[]
    for i in range(len(perm)-1):
        dif = perm[i+1]-perm[i]
        sd.append(int((dif/np.abs(dif) + 1) / 2))
    return tuple(sd)

def genmat(phi, n, nodes):
    phi_inv = [perm_inverse(i) for i in phi]
    A = [0]*(n*n)
    for i in range(n):
        for j in range(n):
            A[n*i + j] = filtrate(tuple(nodes[phi_inv[i]][phi[j]]))
    return A

def symsub(A,n,nodes):
    Aunq = list(set(A))
    s=sp.symbols("x_0:{}".format(len(Aunq)))

    if tuple([1]*(nodes-1)) in Aunq:
        ind = Aunq.index(tuple([1]*(nodes-1)))
        Aunq[ind], Aunq[0] = Aunq[0], Aunq[ind]

    subs = {k: v for k, v in zip(Aunq, s)}
    A = sp.Matrix([subs[i] for i in A]).reshape(n,n)
    return A