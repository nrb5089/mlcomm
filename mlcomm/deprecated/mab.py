import numpy as np


def GetIndex(depth, index_at_depth):
    '''

    :param depth: depth of node in binary tree
    :param index_at_depth: index of node at given depth
    :return: one dimentional index to represent binary tree as array

    As an example the following binary tree
               0
              / \
            1    2
           / \  / \
          3  4 5  6
    is stored in an array as [0,1,2,3,4,5,6]

    The coordinates of node 4 are (2,1) and GetIndex(2,1)=4

    '''

    if (index_at_depth < 0 or index_at_depth > 2 ** depth - 1):
        print("Erroneous indices")
    else:
        return (2 ** depth - 1 + index_at_depth)


def GetIndexAtDepth(index):
    """

    :param index: one dimensional binary tree index
    :return: (depth, index_at_depth)

    This reverses the operation of GetIndex by returning GetIndexAtDepth(4)=(2,1)

    """

    d = int((np.log2(index + 1)))
    return (d, index - 2 ** (d) + 1)


def HOO(GetReward, T=np.Inf, Tmax=1e4, rho=0.5, nu=1, tree_depth=np.Inf):
    '''

    :param tree_depth: if specified, the full tree is allocated; if not, the tree is expanded as needed (inefficient)
    :param rho:
    :param nu:
    :param N: number of arms
    :param T: Horizon of the algorithm; default is None, in which case the algorithm is slightly more efficient
    :param GetReward: function that returns the reward of a given arm
    :return: (reward,pulls)


    '''

    current_index = GetIndex(0, 0)
    current_tree = [current_index]

    reward = []
    pulls = []

    if (np.isinf(tree_depth)):
        current_tree_depth = 1
        qvalues = np.ones(2 ** (current_tree_depth + 1) - 1) * np.Inf
        nplays = np.zeros(2 ** (current_tree_depth + 1) - 1)
        rewards = np.zeros(2 ** (current_tree_depth + 1) - 1)

        pulls = np.zeros(2 ** (current_tree_depth))
    else:
        qvalues = np.ones(2 ** (tree_depth + 1) - 1) * np.Inf
        nplays = np.zeros(2 ** (tree_depth + 1) - 1)
        rewards = np.zeros(2 ** (tree_depth + 1) - 1)

    for t in np.arange(Tmax):
        print("## Round %d" % t)
        print("\tCurrent tree is %s" % str(current_tree))

        current_h = 0
        current_j = 0
        current_index = GetIndex(current_h, current_j)
        current_path = [current_index]

        print("\tStarting path from 0")
        while ((not np.isinf(tree_depth)) and (current_index in current_tree) and (current_h < (tree_depth - 1))) or (
                (np.isinf(tree_depth)) and (current_index in current_tree)):
            (current_h, current_j) = GetIndexAtDepth(current_index)

            if (np.isinf(tree_depth)) and (current_h == current_tree_depth - 1):
                pass  # need to resize tree
                # current_tree_depth = current_tree_depth + 1
                # qvalues.resize(1, 2 ** (current_tree_depth + 1) - 1)  # need to update to inf? * np.Inf
                # nplays.resize(1, 2 ** (current_tree_depth + 1) - 1)
                # rewards.resize(1, 2 ** (current_tree_depth + 1) - 1)

            print("\tQ-value of left child (%d) is %.2f" % (
                GetIndex(current_h + 1, 2 * current_j), qvalues[GetIndex(current_h + 1, 2 * current_j)]))
            print("\tQ-value of right child (%d) is %.2f" % (
                GetIndex(current_h + 1, 2 * current_j + 1), qvalues[GetIndex(current_h + 1, 2 * current_j + 1)]))
            if qvalues[GetIndex(current_h + 1, 2 * current_j)] > qvalues[GetIndex(current_h + 1, 2 * current_j + 1)]:
                current_index = GetIndex(current_h + 1, 2 * current_j)
            elif qvalues[GetIndex(current_h + 1, 2 * current_j)] < qvalues[GetIndex(current_h + 1, 2 * current_j + 1)]:
                current_index = GetIndex(current_h + 1, 2 * current_j + 1)
            else:
                current_index = GetIndex(current_h + 1, 2 * current_j + np.random.randint(0, 2))
            current_path.append(current_index)

            print("\tAdded node %d to path" % current_index)

        print("\tFinal path is %s - Last added node is %d" % (str(current_path), current_index))

        r = GetReward(current_index)
        pulls.append(current_index)
        print("\tObserved reward for %d is %f" % (current_index, r))

        for node in current_path:
            nplays[node] = nplays[node] + 1
            rewards[node] = rewards[node] + r
        print("\tNumber of plays %s" % " ".join(
            x for x in ["{:.0f}".format(nplays[i]) for i in np.arange(nplays.size)]))
        print("\tTotal rewards %s" % " ".join(
            x for x in ["{:.2f}".format(rewards[i]) for i in np.arange(rewards.size)]))

        if current_index not in current_tree:
            current_tree.append(current_index)  # Updating tree
            current_tree.sort()

        tree_hat = current_tree.copy()
        print("\tTree for update %s" % str(tree_hat))

        while (len(tree_hat) > 0):
            leaf = tree_hat.pop()
            (leaf_depth, leaf_index) = GetIndexAtDepth(leaf)

            if np.isinf(T):
                current_q = rewards[leaf] / nplays[leaf] + np.sqrt(
                    2 * np.log(t + 1) / nplays[leaf]) + nu * rho ** leaf_depth
            else:
                current_q = rewards[leaf] / nplays[leaf] + np.sqrt(
                    2 * np.log(T + 1) / nplays[leaf]) + nu * rho ** leaf_depth

            if leaf_depth == tree_depth:
                qvalues[leaf] = current_q
            else:
                child0 = int(GetIndex(leaf_depth + 1, 2 * leaf_index))
                child1 = int(GetIndex(leaf_depth + 1, 2 * leaf_index + 1))
                max_child_q = np.max([qvalues[child0], qvalues[child1]])
                qvalues[leaf] = np.min([max_child_q, current_q])

        print("\tUpdated Qvalues %s" % " ".join(
            x for x in ["{:.2f}".format(qvalues[i]) for i in np.arange(qvalues.size)]))
        print("\tNew tree %s" % str(current_tree))
        print("\n")

        reward.append(r)

    return (reward, pulls)
