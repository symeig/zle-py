#Note: being debugged (use .mma script if you need a gen.py)

import numpy as np
import itertools
from operator import itemgetter
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

def merge_components(comps):
    merged_dict = defaultdict(list)
    for d in comps:
        merged_dict.update(d)
    return merged_dict

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


#transitive reduction
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

def invert_dict_of_lists(d):
    inverted = defaultdict(list)
    for key, values in d.items():
        for value in values:
            inverted[value] = key
    return inverted

def reverse_graph(graph):
    reversed_graph = defaultdict(list)
    for source, targets in graph.items():
        for target in targets:
            reversed_graph[target].append(source)
    return reversed_graph


#block-variable construction
def compute_degrees(graph, node):
    in_degree = 0
    out_degree = len(graph[node])
    
    #returns in/out-degree in order of keys present.
    for key in graph:
        if node in graph[key]:
            in_degree += 1
            
    return [in_degree, out_degree]

def find_max_chain(start, graph, in_out):
    chain = []
    #start node
    chain.append(start)
    if (in_out[start][1] == 1):
        current = graph[start][0]
        #intermediate nodes
        while current in graph and in_out[current][0] == 1 and in_out[current][1] == 1:
            chain.append(current)
            current = graph[current][0]
        #final node
        if (in_out[current][0] == 1):
            chain.append(current)
    return chain

def in_out_dict(graph):   
    in_out = defaultdict(list, {key: [0, 0] for key in list(graph.keys())})
    for node in graph:
        in_out[node][1] = len(graph[node])
        for neighbor in graph[node]:
            in_out[neighbor][0] += 1
            
    return in_out

def contract_graph(graph):
    block_graph = defaultdict(list)
    in_out=in_out_dict(graph)
    g_inds = set(graph.keys())
    chains = []

    #determine block-variables
    while(len(g_inds) != 0):
        chain_i = find_max_chain(list(g_inds)[0],graph, in_out)
        g_inds = set(g_inds) - set(chain_i)
        chains.append(chain_i)

    res=[[all(np.isin(chains[j], i)) for i in chains] for j in range(len(chains))]
    resnew=[]
    for i in range(len(res)):
        if (res[i].count(True) == 1):
            resnew.append(chains[i])

    chains = resnew
    
    #determine block-graph
    block_graph = defaultdict(list)
    block_nodes = [i[0] for i in chains]
    for i in range(len(chains)):
        block_graph[block_nodes[i]] = graph[chains[i][-1]]
        
    return in_out, chains, block_graph


#permutation generation
def comp(setlist):
    set_lengths = [len(i) for i in setlist]
    max_len = np.argmax(set_lengths)
    del set_lengths[max_len]
    next_max = np.argmax(set_lengths)
    return setlist[max_len]-setlist[next_max]
        
#speculative & needs verification for neighbors belonging to 3 or more levels
def nodes_above_below(nodes, levels,filtered_graph_i):
    nodelevels=invert_dict_of_lists(levels) #node/level k/v
    #invert edges
    rg=add_missing_nodes(reverse_graph(filtered_graph_i))
    levelsrg = level_sort(rg) #get its level sort
    nodelevelsrg=invert_dict_of_lists(levelsrg)

    #at and above
    l1ls=[set.union(*[set(i) for i in list(levels.values())[j-1:]]) for j in range(1,np.max(list(levels.keys()))+2)]
    l1l=[len(i) for i in l1ls]
    #at and below
    l2ls=[set.union(*[set(i) for i in list(levelsrg.values())[j-1:]]) for j in range(1,np.max(list(levelsrg.keys()))+2)]
    l2l=[len(i) for i in l2ls]

    above_below={}
    for i in nodes:
        nnodes = filtered_graph_i[i]
        nnodesr = rg[i]
        if(nnodes):
            adj_level = np.min([nodelevels[j] for j in nnodes])
            val = l1l[adj_level]#how many at and above given index
            cumsums=np.unique([l1ls[i] for i in [nodelevels[j] for j in nnodes]])
            if (len(cumsums) >= 2):#individual nodes can belong to multiple levels, distorting the count
                val=len(comp(cumsums))
        else:
            val = 0

        if(nnodesr):
            adj_level = np.min([nodelevelsrg[j] for j in nnodesr])
            valb=l2l[adj_level]#how many at and below given index
            cumsums=np.unique([l2ls[i] for i in [nodelevelsrg[j] for j in nnodesr]])
            if (len(cumsums) >= 2):
                valb=len(comp(cumsums))
        else:
            valb = 0
            
        above_below[i] = [valb, val]
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
    #deals with node sets that aren't 0-based and leaves them in the same order.
    node_map = dict(zip(nodes,np.arange(len(nodes))))#group_nodes[dcompi]
    block_variables=[[list(legal_values[node_map[j]]) for j in i] for i in list(levels.values())]
    ex2=products2(block_variables)
    res = [[elem for s in i for elem in s] for i in list(itertools.product(*ex2))] #these are products of all possibilities
    res = [i for i in res if len(set(i))==len(i)] #these are filtered plausible products
    return res

#2 macro methods for user convenience
def fixed_perms(group_pos, group_nodes):
    #topo sort is used in the transitive reduction
    filtered_graph = [add_missing_nodes(transitive_reduction(j)) for j in [edges_to_adjacency_list(i) for i in group_pos]]
    
    graph_chains = []
    for i in range(len(filtered_graph)):
        in_out, chains, block_graph = contract_graph(filtered_graph[i])
        graph_chains.append(chains)
        
    dc_perms=[]
    for dcompi in range(len(filtered_graph)):
        #print(dcompi)
        nodesi = group_nodes[dcompi]
        n = len(nodesi)
        levels = level_sort(filtered_graph[dcompi])
    
        above_below = nodes_above_below(nodesi,levels, filtered_graph[dcompi])
    
        res = get_perms(n, nodesi,above_below, levels)
        dc_perms.append(res)
    
    legal_values = [group_nodes[0+i[0]:n-i[1]] for i in above_below.values()]
        
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
        print("fixed-order (1..n) permutations: ", len(canonical_perms), canonical_perms[0:3], "\n")
        print("node labels (up to combination): ", len(combination_labels), combination_labels[0:3])
    
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
            A[n*i + j] = filtrate(tuple(nodes[phi_inv[i]][phi[j]])) #remember order of application
            #if (!np.isin(uniques,A[n*i + j])[0]):
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
