import numpy as np

# Disimilarity matrix between the alphabet A,C,G,T
ACGT_DIS = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]


class FastTree(object):
    def __init__(self):
        # Dictionary holding sequences indexed by name
        self.SEQUENCES = {}
        self.PROFILES = {}
        self.CHILDREN = {}
        self.UPDIST = {}
        self.ACTIVE = []
        self.NODENUM = 0
        self.TOTAL_PROFILE = []
        self.ITERATION = 0
        self.VARIANCE_CORR = {}

    def initialize_sequences(self, filename):
        """
        Initialises attributes based on input file.
        :param filename: String
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
        for i in range(int(len(lines) / 2)):
            name = lines[2 * i].strip()[1:]
            sequence = lines[2 * i + 1].strip()
            self.SEQUENCES[name] = sequence
        for seq in self.SEQUENCES:
            self.CHILDREN[seq] = []
            # Updist and variance correction zero for all leaves
            self.UPDIST[seq] = 0
            self.VARIANCE_CORR[seq] = 0
            self.ACTIVE.append(seq)
            #Compute profiles for each of the sequences
            s = self.SEQUENCES[seq]
            freq = np.zeros((4, len(s)))
            for i in range(len(s)):
                if s[i] == 'A':
                    freq[0][i] = 1
                elif s[i] == 'C':
                    freq[1][i] = 1
                elif s[i] == 'G':
                    freq[2][i] = 1
                elif s[i] == 'T':
                    freq[3][i] = 1
            self.PROFILES[seq] = freq

    def update_total_profile(self):
        """
        Calculates the average of all active nodes profiles.
        :returns: 4xL matrix with the average frequency count of each nucleotide over all profiles
        """
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

    def corrected_distances(self, i, j):
        """
        Calculates the corrected distances between two nodes.
        d = -3/4 log(1-4/3 d_u)
        Note: truncated to a maximum of 3
        :param i: node 1
        :param j: node 2
        :returns: corrected distance between nodes i and j
        """
        return min(-3/4 *np.log(1-4/3*self.uncorrected_distance(i,j)),3)

    def compute_variance(self, i, j):
        """
        Calculates the variance used to compute the weights of the joins.
        :param i: node 1
        :param j: node 2
        :returns: variance between nodes i and j
        """
        profile_i, profile_j= self.PROFILES[i], self.PROFILES[j]
        return self.profile_distance(profile_i, profile_j) - self.VARIANCE_CORR[i] - self.VARIANCE_CORR[j]

    def compute_weight(self, i, j, n):
        """
        Calculates the weight of 2 nodes when joining.
        :param i: node 1
        :param j: node 2
        :param n: the number of active nodes before joining
        :returns: the weight of the 2 nodes relative to each other
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

    def relaxed_neighbor_joining(self, A):
        """Find the closest node B to A.
        1. Given node A find closest node B.
        2. Given node B find closest node C
        3. IF: Check if C == A then B is closest to A."""

    def neighbor_join_criterion(self, i, j):
        """
        Get the neighbor join criterion d_u(i,j)-r(i)-r(j) which should be minimized for each join.
        :param i: first node
        :param j: second node
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

    def neighborJoin(self):
        """
        Recursively joins nodes with the lowest neighbour join criterion.
        """

        n = len(self.ACTIVE)
        # recompute the total profile every 200 iterations and at the beginning
        if self.ITERATION % 200 == 0:
            self.update_total_profile()
        self.ITERATION += 1
        # Base case
        if n < 3:
            i, j = self.ACTIVE[0], self.ACTIVE[1]
            self.CHILDREN[(i, j)] = [i, j]
            # (i,j) becomes the root of the tree.
            self.ACTIVE.remove(i), self.ACTIVE.remove(j)
            self.ACTIVE.append((i, j))
            return

        # Find min join criterion
        distances = {(i, j): self.neighbor_join_criterion(i, j) for i in self.ACTIVE for j in self.ACTIVE if i != j}
        new_node = min(distances, key=distances.get)
        i, j = new_node

        weight = self.compute_weight(i, j, n)

        self.CHILDREN[new_node] = [i, j]
        self.PROFILES[new_node] = self.merge_profiles(i, j, weight=weight)
        self.UPDIST[new_node] = self.get_updist(i, j, weight)
        self.VARIANCE_CORR[new_node] = weight * self.VARIANCE_CORR[i] + (1 - weight) * self.VARIANCE_CORR[j] \
                                      + weight * (1 - weight) * self.compute_variance(i, j)
        self.ACTIVE.append(new_node)
        self.ACTIVE.remove(i), self.ACTIVE.remove(j)
        self.TOTAL_PROFILE = (np.array(self.TOTAL_PROFILE)*n - np.array(self.PROFILES[i]) - np.array(self.PROFILES[j])
                              + np.array(self.PROFILES[new_node])) / (n - 1)
        return

    def makeUnRooted(self):
        """
        Unroots rooted tree for the nearest neighbourhood interchange.
        """
        root_child1 = self.ACTIVE[0][0]
        root_child2 = self.ACTIVE[0][1]
        #Join root_child1 and root_child2
        self.CHILDREN[root_child2].append(root_child1)
        self.CHILDREN[root_child1].append(root_child2)
        #Remove root
        del self.CHILDREN[(root_child1,root_child2)]
        #Compute profile of new join
        newProfile = self.merge_profiles(root_child1, root_child2, weight=0.5)
        self.PROFILES[root_child1] = newProfile
        self.PROFILES[root_child2] = newProfile
        return [root_child1,root_child2]

    def recomputeProfiles(self):
        """
        Recomputes profiles for each node in the unrooted tree after each iteration of nearest neighbour interchange.
        """
        for n in self.PROFILES:
            if n not in self.SEQUENCES:
                profiles = [self.PROFILES[x] for x in self.CHILDREN[n]]
                profile = profiles[0]
                for p in profiles[1:]:
                    profile = [[sum(x) for x in zip(profile[i], p[i])] for i in range(4)]
                self.PROFILES[n] = [[t / len(self.CHILDREN[n]) for t in row] for row in profile]

    def nearestNeighbourInterchange(self):
        """
        Runs nearest neighbour interchange on the tree log(N) + 1 times.
        """
        end = int(np.log2(len(self.SEQUENCES)) + 1)
        edges_internal = []
        for i in self.CHILDREN:
            #skips edges coming from root
            if i == self.ACTIVE[0]:
                continue
            for j in self.CHILDREN[i]:
                #skips edges connected to a leaf.
                if j not in self.SEQUENCES:
                    edges_internal.append([i, j])
        #Make tree unrooted, needed to adjust profiles.
        [n1, n2] = self.makeUnRooted()
        #Run nearest neighbour interchange log(N) + 1 times.
        for _ in range(end):
            for i in edges_internal:

                A = self.CHILDREN[i[0]][0]
                B = self.CHILDREN[i[0]][1]
                C = self.CHILDREN[i[1]][0]
                D = self.CHILDREN[i[1]][1]

                # if i[1] child of i[0]
                if i[1] in self.CHILDREN[i[0]]:
                    #if one of the nodes is a child of the root.
                    if len(self.CHILDREN[i[0]]) == 3:
                        if n1 in self.CHILDREN[i[0]]:
                            [A] = [j for j in self.CHILDREN[i[0]] if j !=i[1] and j!=n1]
                            B = n1
                        else:
                            [A] = [j for j in self.CHILDREN[i[0]] if j != i[1] and j != n2]
                            B = n2
                    else:
                        if i[1] == A:
                            A = self.CHILDREN[i[0]][1]
                        for p1 in self.CHILDREN:
                            if i[0] in self.CHILDREN[p1]:
                                B = p1
                # if i[0] child of i[1]
                if i[0] in self.CHILDREN[i[1]]:
                    # if one of the nodes is a child of the root.
                    if len(self.CHILDREN[i[1]]) == 3:
                        if n1 in self.CHILDREN[i[1]]:
                            [C] = [j for j in self.CHILDREN[i[1]] if j !=i[0] and j!=n1]
                            D = n1
                        else:
                            [C] = [j for j in self.CHILDREN[i[1]] if j != i[0] and j != n2]
                            D = n2
                    else:
                        if i[0] == C:
                            C = self.CHILDREN[i[1]][1]
                        for p2 in self.CHILDREN:
                            if i[1] in self.CHILDREN[p2]:
                                D = p2


                dABCD = self.corrected_distances(A,B) + self.corrected_distances(C,D)
                dACBD = self.corrected_distances(A,C) + self.corrected_distances(B,D)
                dADBC = self.corrected_distances(A,D) + self.corrected_distances(B,C)

                # if dABCD < min(dACBD,dADBC):
                #     print("no switch")

                if dACBD < min(dABCD,dADBC):
                    #Switch B and C
                    if B not in self.CHILDREN[i[0]]:
                        self.CHILDREN[i[0]] = [A, C]
                        self.CHILDREN[i[1]] = [D, i[0]]
                        self.CHILDREN[B].remove(i[0])
                        self.CHILDREN[B].append(i[1])
                        edges_internal.remove([B, i[0]])
                        edges_internal.append([B, i[1]])
                    else:
                        self.CHILDREN[i[0]].remove(B)
                        self.CHILDREN[i[1]].remove(C)
                        self.CHILDREN[i[0]].append(C)
                        self.CHILDREN[i[1]].append(B)


                elif dADBC < min(dABCD, dACBD):
                    #Switch B and D
                    if B not in self.CHILDREN[i[0]]:
                        self.CHILDREN[i[0]] = [A, D]
                        self.CHILDREN[i[1]] = [C, i[0]]
                        self.CHILDREN[B].remove(i[0])
                        self.CHILDREN[B].append(i[1])
                        edges_internal.remove([B,i[0]])
                        edges_internal.append([B,i[1]])
                    elif D not in self.CHILDREN[i[1]]:
                        self.CHILDREN[i[0]] = [A, i[1]]
                        self.CHILDREN[i[1]] = [C, B]
                        self.CHILDREN[D].remove(i[1])
                        self.CHILDREN[D].append(i[0])
                        edges_internal.remove([D, i[1]])
                        edges_internal.append([D, i[0]])
                    else:
                        self.CHILDREN[i[0]].remove(B)
                        self.CHILDREN[i[1]].remove(D)
                        self.CHILDREN[i[0]].append(D)
                        self.CHILDREN[i[1]].append(B)

                self.recomputeProfiles()
        #Make tree rooted again before returning.
        self.CHILDREN[(n1, n2)] = [n1, n2]
        self.CHILDREN[n2].remove(n1)
        self.CHILDREN[n1].remove(n2)
        return


    def newickFormat(self,i,str):
        """
        Recursively constructs tree in newick format.
        :param i: current node
        :param str: newick format for ancestors of i.
        :returns: newick format of tree rooted at i.
        """
        if len(self.CHILDREN[i]) == 0:
            return i
        else:
            temp1 = self.newickFormat(self.CHILDREN[i][0],str)
            temp2 = self.newickFormat(self.CHILDREN[i][1],str)
            str = "("+temp1 + ","+temp2+")"
            return str

    # def branchLength(self):
    #     blen = {}
    #     for i in self.CHILDREN:
    #         for j in self.CHILDREN[i]:
    #             if j in self.SEQUENCES:
    #                 blen[j] = self.corrected_distances(j,j)/2
    #             else:
    #                 blen[j] =


