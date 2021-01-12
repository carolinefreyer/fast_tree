import numpy as np
import math

# Disimilarity matrix between the alphabet A,C,G,T
ACGT_DIS = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]


class FastTree(object):
    def __init__(self):
        # Dictionary holding sequences indexed by name
        self.SEQ_NAMES = {}
        self.SEQUENCES = {}
        self.PROFILES = {}
        self.CHILDREN = {}
        self.UPDIST = {}
        self.ACTIVE = []
        self.NODENUM = 0
        self.TOTAL_PROFILE = []
        self.ITERATION = 0
        self.VARIANCE_CORR = {}
        self.TOP_HITS = {}
        self.BEST_JOINS = {}

    def initialize_sequences(self, filename):
        """
        Initialised variables based on input file

        Input: @filename: String
        """

        with open(filename, 'r') as f:
            lines = f.readlines()
        # initialize sequences from file
        for i in range(int(len(lines) / 2)):
            tmp1 = lines[2 * i].strip()[1:]
            tmp2 = lines[2 * i + 1].strip()
            self.SEQ_NAMES[i] = tmp1
            self.SEQUENCES[i] = tmp2
            self.CHILDREN[i] = []
            # Updist and variance correction zero for all leaves
            self.UPDIST[i] = 0
            self.VARIANCE_CORR[i] = 0
            self.ACTIVE.append(i)
            s = self.SEQUENCES[i]
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

    def get_sequence_key(self, value):
        return list(self.SEQUENCES.keys())[list(self.SEQUENCES.values()).index(value)]

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
        Calculates the distances between two profiles, without gaps
        profileDistance(p1,p2) = (1/L)*sum_{i = 0}^L pwd(i)
        where pwd(i) =
                sum(j,k in {A,C,G,T}) freq(p1[i]==j)*freq(p2[i]==k)*ACGT_dis[j][k]

        The profile distance is the average distance between profile characters over all positions.
        Input: @profile1: 4xL matrix profile
               @profile2: 4xL matrix profile
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
        Calculates the uncorrected distance between two profiles, without gaps
        d_u(i,j) = delta(i,j)-upDist(i)-upDist(j)

        Input: @i
               @j
               @profileI: profile of node i
               @profileJ: profile of node j
        """
        profileI, profileJ = self.PROFILES[i], self.PROFILES[j]
        return self.profile_distance(profileI, profileJ) - self.UPDIST[i] - self.UPDIST[j]

    def compute_variance(self, i, j):
        """
        Calculates the variance used to compute the weights of the joins.
        """
        profileI, profileJ = self.PROFILES[i], self.PROFILES[j]
        return self.profile_distance(profileI, profileJ) - self.VARIANCE_CORR[i] - self.VARIANCE_CORR[j]

    def compute_weight(self, i, j, n):
        """
        Calculates the weight of 2 nodes when joining.
        :param i: node 1
        :param j: node 2
        :param n: the number of active nodes before joining
        :return: the weight of the 2 nodes relative to each other
        """
        T = self.TOTAL_PROFILE
        prof_i, prof_j = self.PROFILES[i], self.PROFILES[j]
        # This code will look confusing, but check page 11 of "the better paper" to find the full formula
        numerator = (n - 2) * (self.VARIANCE_CORR[i] - self.VARIANCE_CORR[j]) + n * self.profile_distance(prof_j, T) \
                    - self.get_avg_dist_from_children(j) - n * self.profile_distance(prof_i, T) \
                    + self.get_avg_dist_from_children(i)
        lambd = 0.5 + numerator / (2 * (n - 2) * self.compute_variance(i, j))
        if lambd < 0:
            lambd = 0
        if lambd > 1:
            lambd = 1
        print(lambd)
        return lambd

    def out_distance(self, profile, i):
        """
        Calculates the out-distance for node i
        r(i) = (n*delta(profile,T) - delta(i,i) - (n-1)*upDist(i)-sum_{k =\=i} upDist(k))/(n-2)
        Input: @profile: profile of node i
               @i: node number
               @n: number of active nodes
        """
        T = self.TOTAL_PROFILE
        n = len(self.ACTIVE)
        deltaii = self.get_avg_dist_from_children(i)
        return (n * self.profile_distance(profile, T) - deltaii - (n - 2) * self.UPDIST[i]
                - sum(list(self.UPDIST[x] for x in self.ACTIVE))) / (n - 2)

    def get_avg_dist_from_children(self, i):
        """
        :param i: the node to calculate avg distance of
        :return: the avg distance between node i and its children
        """
        deltaii = 0
        normaliser = 0
        children = self.CHILDREN[i]
        for c1 in children:
            normaliser += 1
            deltaii += self.profile_distance(self.PROFILES[i], self.PROFILES[c1])
        if normaliser != 0:
            deltaii = deltaii / normaliser
        return deltaii

    def merge_profiles(self, seq1, seq2, weight=0.5):
        """
        Calculates the weighted profile of 2 nodes
        :param seq1: the first node
        :param seq2: the second node
        :param weight: the weight of the first node in the join of the 2 nodes. default = 0.5 (unweighted)
        :return: the merged profile of the 2 nodes
        """
        prof1, prof2 = np.array(self.PROFILES[seq1]), np.array(self.PROFILES[seq2])
        prof1 *= weight
        prof2 *= (1 - weight)
        return np.add(prof1, prof2).tolist()

    def neighbor_join_criterion(self, i, j):
        """Get the neighbor join criterion d_u(i,j)-r(i)-r(j) which should be minimized for each join
        :param i, j: node number
        """
        prof_i, prof_j = self.PROFILES[i], self.PROFILES[j]
        return self.uncorrected_distance(i, j) - self.out_distance(prof_i, i) - self.out_distance(prof_j, j)

    def get_updist(self, i, j, weight):
        """
        This method calculates the up-distance after a weighted join of nodes i and j
        :param i: first node that was joined
        :param j: second node that was joined
        :param weight: the weight of node i in the join
        :return: the up-distance of the joined node ij
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
        sorted_distances = sorted(distances.keys(), key=lambda item: distances[item])
        for i in range(m):
            if i >= len(sorted_distances):
                break
            neighbor = sorted_distances[i][1]
            # assign the first element of the top-hits list as best possible join
            if i == 0:
                self.BEST_JOINS.setdefault(num, neighbor)
            # add element to the top hits list
            if num not in self.TOP_HITS:
                self.TOP_HITS.update({num:[neighbor]})
            elif neighbor not in self.TOP_HITS[num]:
                self.TOP_HITS[num].append(neighbor)

    def initialize_top_hits(self):
        """
        Initializes the m top_hits for node A with minimum out distance node A and all nodes within node A's top-hits list.
        The updated top-hits are saved in self.TOP_HITS
        """
        self.update_total_profile()  # Maybe move this to initialize_sequences()
        # 1: select sequence with minimal out distance (and gaps)
        out_distances = {}
        for seq_num in self.ACTIVE:
            out_distances.update({seq_num: self.out_distance(self.PROFILES[seq_num], seq_num)})
        num_A = sorted(out_distances.keys(), key=lambda item: out_distances[item])[0]
        # 2: find top-hits list for node A
        distances_A = {(num_A, j): self.neighbor_join_criterion(num_A, j) for j in self.ACTIVE if num_A != j}
        m = int(math.sqrt(len(self.ACTIVE)))
        self.initialize_nodes_tophits(num_A, distances_A, 2*m)
        # 3: check if restriction du(A,B) <= 0.75 * du(1,H_2m) holds
        if self.uncorrected_distance(num_A, self.TOP_HITS[num_A][0]) <= 0.75 * self.uncorrected_distance(num_A, self.TOP_HITS[num_A][-1]):
            # 4: evaluate top-hits for m neighbours within node A's top-hits
            for i in range(m):
                num_B = self.TOP_HITS[num_A][i]
                distances_B = {(num_B, j): self.neighbor_join_criterion(num_B, j) for j in self.TOP_HITS[num_A] if num_B != j}
                self.initialize_nodes_tophits(num_B, distances_B, m)

    def update_tophits(self, newNode):
        """
        This function compute the top-hits list for the newNode by comparing if to all entries in the top-hits lists of its children.
        :param newNode: Tuple with the two merged sequences
        :return:
        """
        seqA, seqB = self.CHILDREN[newNode]
        # compute the top-hits for the new node
        #top_hitsA, top_hitsB = self.TOP_HITS.get(seqA), self.TOP_HITS.get(seqB)
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
        #self.TOP_HITS.pop(seqA), self.TOP_HITS.pop(seqB)
        m = int(math.sqrt(len(self.ACTIVE)))
        self.initialize_nodes_tophits(newNode, distances, m)
        # self.TOP_HITS.pop(seqA), self.TOP_HITS.pop(seqB)
        # TODO: compare each of the new nodes top-hits to each other
        if newNode not in self.TOP_HITS.keys():
            self.refresh_tophits(newNode)
            return
        for i in self.TOP_HITS[newNode]:
            distances = {(i, j): self.neighbor_join_criterion(i, j) for j in
                         self.TOP_HITS[newNode] if i != j}
            self.initialize_nodes_tophits(i, distances, m)
        # TODO: for all other nodes that either have nodeA or nodeB in their tophits replace with newNode
        replacements = {
            seqA: newNode,
            seqB: newNode
        }
        for key, values in self.TOP_HITS.items():
            self.TOP_HITS[key] = list(set([replacements[x] if x in replacements.keys() else x for x in values]))


    def refresh_tophits(self, newNode):
        """
        If the top hits list are too small this function recomputes the top-hit for the new joined node and
        updates the top-hits lists of the net node's top hits.
        :param newNode:
        :return:
        """
        self.TOP_HITS.clear()
        self.BEST_JOINS.clear()
        # compute the top-hits for the new node
        distances_A = {(newNode, j): self.neighbor_join_criterion(newNode, j) for j in self.ACTIVE if
                       newNode != j}
        m = int(math.sqrt(len(self.ACTIVE)))
        self.initialize_nodes_tophits(newNode, distances_A, 2 * m)
        for i in range(m):
            keyNeighbor = self.TOP_HITS[newNode][i]
            #distances_B = {(keyNeighbor, j): self.neighbor_join_criterion(keyNeighbor, j) for j in
                           #self.TOP_HITS[newNode] if keyNeighbor != j}
            distances_B = {(keyNeighbor, j): self.neighbor_join_criterion(keyNeighbor, j) for j in
                           self.ACTIVE if keyNeighbor != j}
            self.initialize_nodes_tophits(keyNeighbor, distances_B, m)
        self.TOP_HITS[newNode] = self.TOP_HITS[newNode][:m]  # only save the m top hits from the new node

    def update_best_joins(self, i, j):
        copy = dict(self.BEST_JOINS)
        for x in copy:
            if self.BEST_JOINS[x] == i or self.BEST_JOINS[x] == j or x == i or x == j:
                del self.BEST_JOINS[x]

    def newickFormat(self, i, str):
        """
        Recursively constructs tree in newick format.
        :param i: current node
        :param str: newick format for ancestors of i.
        :returns: newick format of tree rooted at i.
        """
        if len(self.CHILDREN[i]) == 0:
            return self.SEQ_NAMES[i]
        else:
            temp1 = self.newickFormat(self.CHILDREN[i][0], str)
            temp2 = self.newickFormat(self.CHILDREN[i][1], str)
            str = "(" + temp1 + "," + temp2 + ")"
            return str

    def neighborJoin(self):

        n = len(self.ACTIVE)
        # update the total profile every 200 iterations and at the beginning
        if self.ITERATION % 200 == 0:
            self.update_total_profile()
        self.ITERATION += 1
        # Base case
        if n < 3:
            # Pia: I don't think it makes sense to calculate the weights, up and out distances for the last join.
            # The formulas don't work with n=2 (division by 0) but we also don't need that info anymore after joining
            # everything. I think.
            n1, n2 = self.ACTIVE[0], self.ACTIVE[1]
            self.CHILDREN[self.NODENUM] = [n1, n2]
            self.ACTIVE.remove(n1), self.ACTIVE.remove(n2)
            self.ACTIVE.append(self.NODENUM)
            self.NODENUM += 1
            return

        # Find min join criterion
        best_join = (None, None)
        best = 0
        for num_A, num_B in self.BEST_JOINS.items():
            join_value = self.neighbor_join_criterion(num_A, num_B)
            if join_value < best:
                best = join_value
                best_join = (num_A, num_B)
        i, j = best_join[0], best_join[1]
        newNode = self.NODENUM
        weight = self.compute_weight(i, j, n)
        self.CHILDREN[newNode] = [i, j]
        self.PROFILES[newNode] = self.merge_profiles(i, j, weight=weight)
        self.UPDIST[newNode] = self.get_updist(i, j, weight)
        self.VARIANCE_CORR[newNode] = weight * self.VARIANCE_CORR[i] + (1 - weight) * self.VARIANCE_CORR[j] \
                                      + weight * (1 - weight) * self.compute_variance(i, j)
        self.ACTIVE.append(newNode)
        self.ACTIVE.remove(i), self.ACTIVE.remove(j)
        self.TOTAL_PROFILE = (np.array(self.TOTAL_PROFILE)*n - np.array(self.PROFILES[i]) - np.array(self.PROFILES[j])
                              + np.array(self.PROFILES[newNode])) / (n - 1)
        # TODO: Place at appropriate position and incoorparate into joining function
        m = int(math.sqrt(len(self.ACTIVE)))
        self.update_best_joins(i,j)
        if len(self.TOP_HITS.keys()) < 0.8 * m: # how to remove already joined nodes if refreshing is not necessary ???
            self.refresh_tophits(newNode)
        else:
            self.update_tophits(newNode)
        self.NODENUM += 1
        return
