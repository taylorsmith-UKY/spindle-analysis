import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform


def compute_dyncut(data, nbClusters, children_map):
    nbVertices = max(children_map)
    inf = float("inf")
    dp = np.zeros((nbClusters + 1, nbVertices + 1)) + inf
    lcut = np.zeros((nbClusters + 1, nbVertices + 1))

    for i in range(0, dp.shape[1]):
        dp[1, i] = compute_intra_variance(i, children_map, data)

    root = max(children_map)
    for vertex in range(len(data), root + 1):
        left_child, right_child = children_map[vertex]
        for k in range(2, nbClusters + 1):
            vmin = inf
            kl_min = -1
            for kl in range(1, k):
                v = dp[kl, left_child] + dp[k - kl, right_child]
                if v < vmin:
                    vmin = v
                    kl_min = kl

            dp[k, vertex] = vmin
            lcut[k, vertex] = kl_min

    return dp, lcut


def build_dict_tree(linkage_matrix):
    tree = {}
    n = linkage_matrix.shape[0] + 1
    for i in range(0, n - 1):
        tree[linkage_matrix[i, 0]] = n + i
        tree[linkage_matrix[i, 1]] = n + i
    return tree


def build_children_map(tree):
    children_map = {}
    for k, v in tree.items():
        children_map[v] = children_map.get(v, [])
        children_map[v].append(int(k))
    return children_map


def build_children(vertex, children_map):
    children = []
    if vertex in children_map:
        left_child, right_child = children_map[vertex]
        if left_child in children_map:
            children.extend(build_children(left_child, children_map))
        else:
            children.extend([left_child])

        if right_child in children_map:
            children.extend(build_children(right_child, children_map))
        else:
            children.extend([right_child])

    return children


def get_var(data, subpop):
    intravar = 0
    center = np.mean(data[subpop], axis=0)
    for elem in subpop:
        x = data[elem] - center
        intravar += np.dot(x, x)
    return intravar


def compute_intra_variance(vertex, children_map, data):
    children = build_children(vertex, children_map)
    intravar = 0
    if children:
        intravar = get_var(data, children)

    return intravar


def compute_centers(data, target):
    centers = []
    for i in set(target):
        id_pts = [index for index, value in enumerate(target) if value == i]
        centers.append(np.mean(data[id_pts], axis=0))

    return centers


def compute_flat_dyn_clusters(cur_vertex, k, lcut, children_map):
    clusters = []
    # leaf
    if k == 1 and not cur_vertex in children_map:
        clusters.append([cur_vertex])
    # one cluster left, get the leaves
    if k == 1 and cur_vertex in children_map:
        leaves = build_children(cur_vertex, children_map)
        clusters.append(leaves)
    # recurse in left and right subtrees
    if k > 1:
        if cur_vertex in children_map:
            left_child, right_child = children_map[cur_vertex]
            clusters.extend(compute_flat_dyn_clusters(left_child, int(lcut[k, cur_vertex]), lcut, children_map))
            clusters.extend(compute_flat_dyn_clusters(right_child, int(k - lcut[k, cur_vertex]), lcut, children_map))

    return clusters


def compute_flat_cut_clusters(nbClusters, linkage_matrix):
    flat = fcluster(linkage_matrix, nbClusters, 'maxclust')
    flat_clusters = []
    for i in range(1, len(set(flat)) + 1):
        flat_clusters.append([index for index, value in enumerate(flat) if value == i])

    return flat_clusters


def bench_methods(data, nbClusters, methods, fname=''):
    d = pdist(data)
    if fname != '':
        f = open(fname, 'w')
    for method in methods:
        if method in ['centroid', 'ward', 'median']:
            linkage_matrix = linkage(data, method)
        else:
            linkage_matrix = linkage(d, method)
        tree = build_dict_tree(linkage_matrix)
        children_map = build_children_map(tree)
        dp, lcut = compute_dyncut(data, nbClusters, children_map)
        flat_dyn_clusters = compute_flat_dyn_clusters(max(children_map), nbClusters, lcut, children_map)
        flat_cut_clusters = compute_flat_cut_clusters(nbClusters, linkage_matrix)

        tot_dyn = 0
        tot_cut = 0
        for i in range(0, nbClusters):
            tot_dyn += get_var(data, flat_dyn_clusters[i])
            tot_cut += get_var(data, flat_cut_clusters[i])
        if fname == '':
            print("method: %s" % method)
            print("intra-variance: (DP): %.4f,\t(cst height)%.4f\n\n" % (tot_dyn, tot_cut))
        else:
            f.write("method: %s\n" % method)
            f.write("intra-variance: (DP): %.4f,\t(cst height)%.4f\n\n" % (tot_dyn, tot_cut))
    if fname != '':
        f.close()
