import numpy as np
import math

# Disimilarity matrix between the alphabet A,C,G,T
ACGT_DIS = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]


class FastTree(object):
    def __init__(self):

        self.NODENUM = 0
        self.ITERATION = 0

        self.SEQ_NAMES = {} # Dictionary holding sequence's named indexed by node number
        self.SEQUENCES = {} # Dictionary holding sequences indexed by node number
        self.PROFILES = {} # Dictionary holding profiles indexed by node number
        self.CHILDREN = {} # Dictionary holding children indexed by node number
        self.UPDIST = {} # Dictionary holding up-distance indexed by node number
        self.ACTIVE = [] # List holding node number of active nodes
        self.TOTAL_PROFILE = [] # Total profile of the tree
        self.VARIANCE_CORR = {} # Dictionary holding variance correlation indexed by node number
        self.TOP_HITS = {} # Dictionary holding top hits indexed by node number
        self.BEST_JOINS = {} # Dictionary holding best joins indexed by node number
        self.BRANCH_LENGTHS = {} # Dictionary holding branch lengths indexed by node number
        self.AGES = {} # Dictionary holding ages indexed by node number

    def initialize_sequences(self, filename):
        """
        Initialises attributes based on input file.
        :param filename: String
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
        # initialize sequences from file
        for i in range(int(len(lines) / 2)):
            name = lines[2 * i].strip()[1:]
            sequence = lines[2 * i + 1].strip()
            self.SEQ_NAMES[i] = name
            self.SEQUENCES[i] = sequence
            self.CHILDREN[i] = []
            # Updist and variance correction zero for all leaves
            self.UPDIST[i] = 0
            self.VARIANCE_CORR[i] = 0
            self.AGES[i] = 0
            #sequences become active immediately
            self.ACTIVE.append(i)
            self.BRANCH_LENGTHS[i] = 1
            s = self.SEQUENCES[i]
            #Compute profile of sequence
            freq = np.zeros((4, len(s)))
            for j in range(len(s)):
                if s[j] == 'A':
                    freq[0][j] = 1
                elif s[j] == 'C':
                    freq[1][j] = 1
                elif s[j] == 'G':
                    freq[2][j] = 1
                elif s[j] == 'T':
                    freq[3][j] = 1
            self.PROFILES[i] = freq
            self.NODENUM += 1
        self.update_total_profile()

    def update_total_profile(self):
        """Calculates the average of all active nodes profiles
        :return: 4xL matrix with the average frequency count of each nucleotide over all profiles"""
        seqs = self.ACTIVE
        L = len(seqs)
        profiles = [self.PROFILES[x] for x in seqs]
        T = profiles[0]
        for p in profiles[1:]:
            T = [[sum(x) for x in zip(T[i], p[i])] for i in range(4)]
        T = [[t / L for t in row] for row in T]
        self.TOTAL_PROFILE = T

    def profile_distance(self, profile1, profile2):
        """
        Calculates the distances between two profiles, without gaps.
        profileDistance(p1,p2) = (1/L)*sum_{i = 0}^L pwd(i)
        where pwd(i) =
                sum(j,k in {A,C,G,T}) freq(p1[i]==j)*freq(p2[i]==k)*ACGT_dis[j][k]
        The profile distance is the average distance between profile characters over all positions.
        :param profile1: 4xL matrix profile
        :param profile2: 4xL matrix profile
        :returns: profile distance between profile_1 and profile_2
        """
        d = 0
        # Length of sequence
        L = len(profile1[0])
        for i in range(L):
            for j in range(4):
                for k in range(4):
                    d += (profile1[j][i] * profile2[k][i] * ACGT_DIS[j][k])
        return d / L

    def uncorrected_distance(self, i, j):
        """
        Calculates the uncorrected distance between two profiles, without gaps.
        d_u(i,j) = delta(i,j)-upDist(i)-upDist(j)

        :param i: node 1
        :param j: node 2
        :returns: uncorrected distance between nodes i and j
        """
        profile_i, profile_j = self.PROFILES[i], self.PROFILES[j]
        return self.profile_distance(profile_i, profile_j) - self.UPDIST[i] - self.UPDIST[j]

    def corrected_distance(self, i, j):
        """
        Calculates the corrected distances between two nodes.
        d = -3/4 log(1-4/3 d_u)
        Note: truncated to a maximum of 3
        :param i: node 1
        :param j: node 2
        :returns: corrected distance between nodes i and j
        """
        return min(-3 / 4 * np.log(1 - 4 / 3 * self.uncorrected_distance(i, j)), 3)

    def compute_variance(self, i, j):
        """
        Calculates the variance used to compute the weights of the joins.
        :param i: node 1
        :param j: node 2
        :returns: variance between nodes i and j
        """
        profile_i, profile_j = self.PROFILES[i], self.PROFILES[j]
        return self.profile_distance(profile_i, profile_j) - self.VARIANCE_CORR[i] - self.VARIANCE_CORR[j]

    def compute_weight(self, i, j, n):
        """
        Calculates the weight of node i in the join (i,j).
        :param i: node 1
        :param j: node 2
        :param n: the number of active nodes before joining
        :returns: the weight of the 2 nodes relative to each other
        """
        T = self.TOTAL_PROFILE
        prof_i, prof_j = self.PROFILES[i], self.PROFILES[j]
        numerator = (n - 2) * (self.VARIANCE_CORR[i] - self.VARIANCE_CORR[j]) + n * self.profile_distance(prof_j, T) \
                    - self.get_avg_dist_from_children(j) - n * self.profile_distance(prof_i, T) \
                    + self.get_avg_dist_from_children(i)
        lambd = 0.5 + numerator / (2 * (n - 2) * self.compute_variance(i, j))
        if lambd < 0:
            lambd = 0
        if lambd > 1:
            lambd = 1
        return lambd

    def out_distance(self, profile, i):
        """
        Calculates the out-distance for node i.
        r(i) = (n*delta(profile,T) - delta(i,i) - (n-1)*upDist(i)-sum_{k =\=i} upDist(k))/(n-2)
        :param profile: profile of node i
        :param i: node
        :returns: out distance of node i
        """
        T = self.TOTAL_PROFILE
        n = len(self.ACTIVE)
        delta_ii = self.get_avg_dist_from_children(i)
        return (n * self.profile_distance(profile, T) - delta_ii - (n - 2) * self.UPDIST[i]
                - sum(list(self.UPDIST[x] for x in self.ACTIVE))) / (n - 2)

    def get_avg_dist_from_children(self, i):
        """
        Computes the average distance between the node i and its children.
        :param i: the node to calculate avg distance of
        :returns: the average distance between node i and its children
        """
        delta_ii = 0
        normaliser = 0
        children = self.CHILDREN[i]
        for c in children:
            normaliser += 1
            delta_ii += self.profile_distance(self.PROFILES[i], self.PROFILES[c])
        if normaliser != 0:
            delta_ii = delta_ii / normaliser
        return delta_ii

    def merge_profiles(self, seq1, seq2, weight=0.5):
        """
        Calculates the weighted profile of 2 nodes.
        :param seq1: the first node
        :param seq2: the second node
        :param weight: the weight of the first node in the join of the 2 nodes. default = 0.5 (unweighted)
        :returns: the merged profile of the 2 nodes
        """
        prof1, prof2 = np.array(self.PROFILES[seq1]), np.array(self.PROFILES[seq2])
        prof1 *= weight
        prof2 *= (1 - weight)
        return np.add(prof1, prof2).tolist()

    def neighbor_join_criterion(self, i, j):
        """Get the neighbor join criterion d_u(i,j)-r(i)-r(j) which should be minimized for each join
        :param i, j: node number
        :returns: the neighbour join criterion for nodes i and j
        """
        prof_i, prof_j = self.PROFILES[i], self.PROFILES[j]
        return self.uncorrected_distance(i, j) - self.out_distance(prof_i, i) - self.out_distance(prof_j, j)

    def get_updist(self, i, j, weight):
        """
        This method calculates the up-distance after a weighted join of nodes i and j.
        :param i: first node that was joined
        :param j: second node that was joined
        :param weight: the weight of node i in the join
        :returns: the up-distance of the joined node ij
        """
        out_i = self.out_distance(self.PROFILES[i], i)
        out_j = self.out_distance(self.PROFILES[j], j)
        du_i_ij = (self.uncorrected_distance(i, j) + out_i - out_j) / 2
        du_j_ij = (self.uncorrected_distance(i, j) + out_j - out_i) / 2
        u_ij = weight * (self.UPDIST[i] + du_i_ij) + (1 - weight) * (self.UPDIST[j] + du_j_ij)
        return u_ij

    def initialize_nodes_tophits(self, num, distances, m):
        """
        Updates self.TOP_HITS with the m top-hits for the node given its closest neighbours in sorted_distances
        :param num: node number
        :param distances: dictionary with a key tuple (node, neighbor) and the distance as value
        :param m: number of top-hits to be initialized
        """
        if num in self.TOP_HITS:
            del self.TOP_HITS[num]
        sorted_distances = sorted(distances.keys(), key=lambda item: distances[item])
        for i in range(m):
            if i >= len(sorted_distances):
                break
            neighbor = sorted_distances[i][1]
            # assign the first element of the top-hits list as best possible join
            if i == 0:
                self.BEST_JOINS[num] = (neighbor, distances[(num,neighbor)])
            # check if the distance to this top-hit is smaller than this top-hits best join
            if neighbor not in self.BEST_JOINS or self.BEST_JOINS[neighbor][1] > distances[(num,neighbor)]:
                self.BEST_JOINS[neighbor] = (num, distances[(num,neighbor)])
            # add element to the top hits list
            if num not in self.TOP_HITS:
                self.TOP_HITS.update({num: [neighbor]})
            elif neighbor not in self.TOP_HITS[num]:
                self.TOP_HITS[num].append(neighbor)

    def initialize_top_hits(self):
        """
        Initializes the m top_hits for each node, starting with the nodes with the smallest out-distance.
        The updated top-hits are saved in self.TOP_HITS
        """
        # 1: select sequence with minimal out distance (and gaps)
        out_distances = {}
        for seq_num in self.ACTIVE:
            out_distances.update({seq_num: self.out_distance(self.PROFILES[seq_num], seq_num)})
        # all nodes sorted by out distance
        srt = sorted(out_distances.keys(), key=lambda item: out_distances[item])
        m = int(math.sqrt(len(self.ACTIVE)))
        while len(srt) > 0:
            num_A = srt[0]
            # 2: find top-hits list for node A
            distances_A = {(num_A, j): self.neighbor_join_criterion(num_A, j) for j in self.ACTIVE if num_A != j}
            self.initialize_nodes_tophits(num_A, distances_A, m)
            distances_A = [x[1] for x in sorted(distances_A.keys(), key=lambda item: distances_A[item])]
            # 3: check if restriction du(A,B) <= 0.75 * du(1,H_2m) holds
            for num_B in self.TOP_HITS[num_A]:
                if self.uncorrected_distance(num_A, num_B) <= 0.75 * self.uncorrected_distance(num_A,
                                                                                               distances_A[2 * m - 1]):
                    # 4: evaluate top-hits for m neighbours within node A's 2m top-hits
                    distances_B = {(num_B, j): self.neighbor_join_criterion(num_B, j) for j in distances_A[:2 * m] if
                                   num_B != j}
                    self.initialize_nodes_tophits(num_B, distances_B, m)
            srt = srt[1:]


    def update_tophits(self, newNode, m):
        """
        This function computes the top-hits list for the newNode by comparing if to all entries in the top-hits lists of its children.
        :param newNode: Tuple with the two merged sequences
        :param m : the amount of top-hits that will be saved. This is set to sqrt(n).
        """
        seqA, seqB = self.CHILDREN[newNode]
        # compute the top-hits for the new node
        if seqA in self.TOP_HITS.keys():
            top_hitsA = self.TOP_HITS.get(seqA) if self.TOP_HITS.get(seqA) != None else []
            self.TOP_HITS.pop(seqA)
        else:
            top_hitsA = []
        if seqB in self.TOP_HITS.keys():
            top_hitsB = self.TOP_HITS.get(seqB) if self.TOP_HITS.get(seqB) != None else []
            self.TOP_HITS.pop(seqB)
        else:
            top_hitsB = []
        candidates = [i for i in set(top_hitsA + top_hitsB) if i != seqA and i != seqB]  # remove duplicates
        distances = {(newNode, j): self.neighbor_join_criterion(newNode, j) for j in candidates}
        self.initialize_nodes_tophits(newNode, distances, m)

        replacements = {
            seqA: newNode,
            seqB: newNode
        }
        for key, values in self.TOP_HITS.items():
            self.TOP_HITS[key] = list(set([replacements[x] if x in replacements.keys() else x for x in values]))


    def refresh_tophits(self, newNode, m):
        """
        If the top hits list are too small for the join connected to newNode or if the age of newNode (last time
        since a refresh) exceeds a threshold. This function recomputes the top-hit for the new joined node and
        updates the top-hits lists of the new node's top hits.
        :param newNode: the join whose top-hits are to be refreshed
        :param m : the amount of top-hits that will be saved. This is set to sqrt(n).
        """
        # set the age of the node to 0
        self.AGES[newNode] = 0
        # compute the top-hits for the new node
        distances_A = {(newNode, j): self.neighbor_join_criterion(newNode, j) for j in self.ACTIVE if
                       newNode != j}
        self.initialize_nodes_tophits(newNode, distances_A, m)
        distances_A = [x[1] for x in sorted(distances_A.keys(), key=lambda item: distances_A[item])]
        # recalculate the top hits of the new node's top hits by comparing it to the new node's top 2m hits.
        for num_B in self.TOP_HITS[newNode]:
            distances_B = {(num_B, j): self.neighbor_join_criterion(num_B, j) for j in distances_A[:2 * m] if
                           num_B != j}
            self.initialize_nodes_tophits(num_B, distances_B, m)
            self.AGES[num_B] = 0

    def update_best_joins(self, i, j):
        """
        This method ensures that the best joins only point to active nodes.
        :param i: the first child of a new join
        :param j: the second child of a new join
        """
        copy = dict(self.BEST_JOINS)
        for x in copy:
            if i in copy[x] or j in copy[x] or x == i or x == j:
                del self.BEST_JOINS[x]

    def neighbor_join(self):
        """
        Recursively joins nodes with the lowest neighbour join criterion using the top-hits and best-joins lists.
        """
        n = len(self.ACTIVE)
        #length of top hits
        m = int(math.sqrt(len(self.SEQUENCES)))
        # recompute the total profile every 200 iterations and at the beginning
        if self.ITERATION % 200 == 0:
            self.update_total_profile()
        self.ITERATION += 1
        # Base case
        if n < 3:
            i, j = self.ACTIVE[0], self.ACTIVE[1]
            self.CHILDREN[self.NODENUM] = [i, j]
            self.ACTIVE.remove(i), self.ACTIVE.remove(j)
            self.PROFILES[self.NODENUM] = self.merge_profiles(i, j, 0.5)
            self.UPDIST[self.NODENUM] = 0.5 * (self.UPDIST[i] + self.UPDIST[j]) + self.uncorrected_distance(i, j)
            self.ACTIVE.append(self.NODENUM)
            self.NODENUM += 1
            return

        # Find min join criterion by recalculating the neighbor join criterion of the top m best joins.
        best_join = (None, None)
        best = 0
        top_m_best_joins = sorted(self.BEST_JOINS.items(), key=lambda x: x[1][1])[:m]
        for num_A, [num_B, _] in top_m_best_joins:
            join_value = self.neighbor_join_criterion(num_A, num_B)
            if join_value < best:
                best = join_value
                best_join = (num_A, num_B)
        i, j = best_join[0], best_join[1]

        # Apply relaxed neighbor joining by checking if there are no better joins in the top-hits of the selected nodes.
        for hit in self.TOP_HITS[i]:
            join_value = self.neighbor_join_criterion(hit, i)
            if join_value < best:
                best = join_value
                best_join = (i, hit)
        for hit in self.TOP_HITS[j]:
            join_value = self.neighbor_join_criterion(hit, j)
            if join_value < best:
                best = join_value
                best_join = (j, hit)

        i, j = best_join[0], best_join[1]

        new_node = self.NODENUM
        weight = self.compute_weight(i, j, n)

        self.CHILDREN[new_node] = [i, j]
        self.PROFILES[new_node] = self.merge_profiles(i, j, weight=weight)
        self.UPDIST[new_node] = self.get_updist(i, j, weight)
        self.VARIANCE_CORR[new_node] = weight * self.VARIANCE_CORR[i] + (1 - weight) * self.VARIANCE_CORR[j] \
                                       + weight * (1 - weight) * self.compute_variance(i, j)
        self.AGES[new_node] = 1 + max(self.AGES[i], self.AGES[j])
        self.ACTIVE.append(new_node)
        self.ACTIVE.remove(i), self.ACTIVE.remove(j)
        self.TOTAL_PROFILE = (np.array(self.TOTAL_PROFILE) * n - np.array(self.PROFILES[i]) - np.array(self.PROFILES[j])
                              + np.array(self.PROFILES[new_node])) / (n - 1)

        # calculate the top hits of the new node
        if len(self.ACTIVE) > 2:
            self.update_best_joins(i, j)
            self.update_tophits(new_node, m)
            if len(self.TOP_HITS) < 0.8 * m or self.AGES[new_node] > 1 + np.log(m):
                self.refresh_tophits(new_node, m)
        self.BRANCH_LENGTHS[new_node] = 1
        self.NODENUM += 1

        return

    def compute_weight_NNI(self, A , B, C, D):
        weight = 1 / 2 + (self.corrected_distance(B, C) + self.corrected_distance(B, D) -
                          self.corrected_distance(A, C) - self.corrected_distance(A, D)) / (
                             4 * self.corrected_distance(A, B))
        if weight < 0:
            weight = 0
        if weight > 1:
            weight = 1
        return weight

    def recompute_profiles(self, join1, A, B, join2, C, D):
        weight_AB = self.compute_weight_NNI(A, B, C, D)
        self.PROFILES[join1] = self.merge_profiles(A, B, weight=weight_AB)
        weight_CD = self.compute_weight_NNI(C, D, A, B)
        self.PROFILES[join2] = self.merge_profiles(C, D, weight=weight_CD)

    def get_internal_edges(self):
        # Find internal edges
        edges_internal = []
        # Recurses through Children starting from leafs so that children are visited before parents.
        for i in self.CHILDREN:
            # skips edges coming from root
            if i != self.ACTIVE[0]:
                for j in self.CHILDREN[i]:
                    # skips edges connected to a leaf.
                    if j not in self.SEQUENCES:
                        edges_internal.append({i, j})
        return edges_internal


    def nearest_neighbour_interchange(self):
        """
        Runs nearest neighbour interchange on the tree log(N) + 1 times.
        """

        root = self.ACTIVE[0]
        root_child1, root_child2 = self.CHILDREN[root]
        del self.CHILDREN[root]

        end = int(np.log2(len(self.SEQUENCES)) + 1)
        # Run nearest neighbour interchange log(N) + 1 times.
        for _ in range(end):
            edges_internal = self.get_internal_edges()
            for e in edges_internal:
                i = list(e)
                A, B = self.CHILDREN[i[0]]
                C, D = self.CHILDREN[i[1]]

                # if i[1] child of i[0]
                if i[1] in self.CHILDREN[i[0]]:
                    [A] = [j for j in self.CHILDREN[i[0]] if j != i[1]]
                    if i[0] == root_child1:
                        B = root_child2
                    elif i[0] == root_child2:
                        B = root_child1
                    else:
                        [B] = [p1 for p1 in self.CHILDREN if i[0] in self.CHILDREN[p1]]

                # if i[0] child of i[1]
                if i[0] in self.CHILDREN[i[1]]:
                    [C] = [j for j in self.CHILDREN[i[1]] if j != i[0]]
                    if i[1] == root_child1:
                        D = root_child2
                    elif i[1] == root_child2:
                        D = root_child1
                    else:
                        [D] = [p2 for p2 in self.CHILDREN if i[1] in self.CHILDREN[p2]]

                dABCD = self.corrected_distance(A, B) + self.corrected_distance(C, D)
                dACBD = self.corrected_distance(A, C) + self.corrected_distance(B, D)
                dADBC = self.corrected_distance(A, D) + self.corrected_distance(B, C)

                if dABCD < min(dACBD, dADBC):
                    print("no switch")
                    if sorted([i[0], B]) == sorted([root_child1, root_child2]):
                        self.recompute_profiles(i[0], A, i[1], i[1], C, D)
                    elif sorted([i[1], D]) == sorted([root_child1, root_child2]):
                        self.recompute_profiles(i[0], A, B, i[1], C, i[0])
                    else:
                        self.recompute_profiles(i[0], A, B, i[1], C, D)

                if dACBD < min(dABCD, dADBC):
                    # Switch B and C
                    if B not in self.CHILDREN[i[0]]:
                        self.CHILDREN[i[0]] = [A, C]
                        self.CHILDREN[i[1]] = [D, i[0]]
                        if sorted([i[0], B]) != sorted([root_child1, root_child2]):
                            self.CHILDREN[B].remove(i[0])
                            self.CHILDREN[B].append(i[1])
                            self.recompute_profiles(i[0], A, C, i[1], B, D)
                        else:
                            self.recompute_profiles(i[0], A, C, i[1], i[0], D)
                        if i[0] == root_child1:
                            root_child1 = i[1]
                        if i[0] == root_child2:
                            root_child2 = i[1]
                    else:
                        self.CHILDREN[i[0]].remove(B)
                        self.CHILDREN[i[1]].remove(C)
                        self.CHILDREN[i[0]].append(C)
                        self.CHILDREN[i[1]].append(B)
                        if sorted([i[1], D]) == sorted([root_child1, root_child2]):
                            self.recompute_profiles(i[0], A, C, i[1], B, i[0])
                        else:
                            self.recompute_profiles(i[0], A, C, i[1], B, D)
                    if {i[0], B} in edges_internal:
                        edges_internal.remove({i[0], B})
                    if {i[1], C} in edges_internal:
                        edges_internal.remove({i[1], C})


                if dADBC < min(dABCD, dACBD):
                    # Switch B and D
                    print("Switch B and D")
                    if B not in self.CHILDREN[i[0]]:
                        self.CHILDREN[i[0]] = [A, D]
                        self.CHILDREN[i[1]] = [C, i[0]]
                        if sorted([i[0], B]) != sorted([root_child1,root_child2]):
                            self.CHILDREN[B].remove(i[0])
                            self.CHILDREN[B].append(i[1])
                            self.recompute_profiles(i[0], A, D, i[1], C, B)
                        else:
                            self.recompute_profiles(i[0], A, D, i[1], C, i[0])
                        if i[0] == root_child1:
                            root_child1 = i[1]
                        if i[0] == root_child2:
                            root_child2 = i[1]
                        if {i[0],B} in edges_internal:
                            edges_internal.remove({i[0],B})
                        if {i[1],D} in edges_internal:
                            edges_internal.remove({i[1],D})
                    elif D not in self.CHILDREN[i[1]]:
                        self.CHILDREN[i[0]] = [A, i[1]]
                        self.CHILDREN[i[1]] = [C, B]
                        if sorted([i[1], D]) != sorted([root_child1,root_child2]):
                            self.CHILDREN[D].remove(i[1])
                            self.CHILDREN[D].append(i[0])
                            self.recompute_profiles(i[0], A, D, i[1], C, B)
                        else:
                            self.recompute_profiles(i[0], A, i[1], i[1], C, B)
                        if i[1] == root_child1:
                            root_child1 = i[0]
                        if i[1] == root_child2:
                            root_child2 = i[0]
                    else:
                        self.CHILDREN[i[0]].remove(B)
                        self.CHILDREN[i[1]].remove(D)
                        self.CHILDREN[i[0]].append(D)
                        self.CHILDREN[i[1]].append(B)
                        self.recompute_profiles(i[0], A, D, i[1], C, B)
                    if {i[0], B} in edges_internal:
                        edges_internal.remove({i[0], B})
                    if {i[1], D} in edges_internal:
                        edges_internal.remove({i[1], D})

        self.CHILDREN[root] = [root_child1,root_child2]
        return

    def recompute_profiles_branch_lengths(self):
        n = len(self.SEQUENCES)
        for i in self.CHILDREN:
            if len(self.CHILDREN[i]) != 0:
                if n != 2:
                    weight = self.compute_weight(self.CHILDREN[i][0], self.CHILDREN[i][1], n)
                else:
                    weight = 0.5
                self.PROFILES[i] = self.merge_profiles(self.CHILDREN[i][0], self.CHILDREN[i][1], weight=weight)
                n -= 1

    def newick_format(self, i, res):
        """
        Recursively constructs tree in newick format.
        :param i: current node
        :param str: newick format for ancestors of i.
        :returns: newick format of tree rooted at i.
        """
        if len(self.CHILDREN[i]) == 0:
            return i
        else:
            temp1 = self.newick_format(self.CHILDREN[i][0], res)
            temp2 = self.newick_format(self.CHILDREN[i][1], res)
            if type(temp1) is int:
                temp1 = self.SEQ_NAMES[temp1] + ":" + str(self.BRANCH_LENGTHS[self.CHILDREN[i][0]])
            if type(temp2) is int:
                temp2 = self.SEQ_NAMES[temp2] + ":" + str(self.BRANCH_LENGTHS[self.CHILDREN[i][1]])
            if i != self.ACTIVE[0]:
                res = "(" + temp1 + "," + temp2 + "):" + str(self.BRANCH_LENGTHS[i])
            else:
                res = "(" + temp1 + "," + temp2 + ")"
            return res

    def find_family_leaf(self, node):
        for p in self.CHILDREN:
            if node in self.CHILDREN[p]:
                parent = p
        for g in self.CHILDREN:
            if parent in self.CHILDREN[g]:
                grandparent = g
        if node == self.CHILDREN[parent][0]:
            sibling = self.CHILDREN[parent][1]
        else:
            sibling = self.CHILDREN[parent][0]
        return sibling, grandparent

    def find_family_internal(self, node):
        [child1, child2] = self.CHILDREN[node]
        for p in self.CHILDREN:
            if node in self.CHILDREN[p]:
                parent = p
        if node == self.CHILDREN[parent][0]:
            sibling = self.CHILDREN[parent][1]
        else:
            sibling = self.CHILDREN[parent][0]
        return child1, child2, parent, sibling

    def compute_branch_lenghts(self):
        self.recompute_profiles_branch_lengths()
        root = self.ACTIVE[0]
        self.iter_branch_length(self.CHILDREN[root][0])
        self.iter_branch_length(self.CHILDREN[root][1])

    def iter_branch_length(self, node):
        if len(self.CHILDREN[node]) == 0:
            # leaf
            sibling, grandparent = self.find_family_leaf(node)
            d_i_g = self.corrected_distance(node, grandparent)
            d_i_s = self.corrected_distance(node, sibling)
            d_s_g = self.corrected_distance(sibling, grandparent)
            self.BRANCH_LENGTHS[node] = round((d_i_g + d_i_s - d_s_g) / 2, 3)
        else:
            # internal
            child1, child2, parent, sibling = self.find_family_internal(node)
            d_c1_p = self.corrected_distance(child1, parent)
            d_c2_p = self.corrected_distance(child2, parent)
            d_c1_s = self.corrected_distance(child1, sibling)
            d_c2_s = self.corrected_distance(child2, sibling)
            d_c1_c2 = self.corrected_distance(child1, child2)
            d_s_p = self.corrected_distance(sibling, parent)
            self.BRANCH_LENGTHS[node] = round((d_c1_p + d_c2_p + d_c1_s + d_c2_s) / 4 - (d_c1_c2 + d_s_p) / 2, 3)
            self.iter_branch_length(child1)
            self.iter_branch_length(child2)
