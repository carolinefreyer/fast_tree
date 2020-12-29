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
        Initialised variables based on input file

        Input: @filename: String
        """

        with open(filename, 'r') as f:
            lines = f.readlines()
        for i in range(int(len(lines) / 2)):
            tmp1 = lines[2 * i].strip()[1:]
            tmp2 = lines[2 * i + 1].strip()
            self.SEQUENCES[tmp1] = tmp2
        for seq in self.SEQUENCES:
            self.CHILDREN[seq] = []
            # Updist and variance correction zero for all leaves
            self.UPDIST[seq] = 0
            self.VARIANCE_CORR[seq] = 0
            self.ACTIVE.append(seq)
            freq = np.zeros((4, len(self.SEQUENCES[seq])))
            for i in range(len(self.SEQUENCES[seq])):
                if self.SEQUENCES[seq][i] == 'A':
                    freq[0][i] = 1
                elif self.SEQUENCES[seq][i] == 'C':
                    freq[1][i] = 1
                elif self.SEQUENCES[seq][i] == 'G':
                    freq[2][i] = 1
                elif self.SEQUENCES[seq][i] == 'T':
                    freq[3][i] = 1
            self.PROFILES[seq] = freq
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
        :param profile1: 4xL matrix profile
        :param profile2: 4xL matrix profile
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

        :param i: node 1
        :param j: node 2
        """
        profileI, profileJ = self.PROFILES[i], self.PROFILES[j]
        return self.profile_distance(profileI, profileJ) - self.UPDIST[i] - self.UPDIST[j]

    def corrected_distances(self, i, j):
        """
        Calculates the corrected distances between two nodes
        d = -3/4 log(1-4/3 d_u)
        Note: truncated to a maximum of 3

        :param i: node 1
        :param j: node 2
        """
        return min(-3/4 *np.log(1-4/3*self.uncorrected_distance(i,j)),3)

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
        ans = (n * self.profile_distance(profile, T) - deltaii - (n - 2) * self.UPDIST[i]
                - sum(list(self.UPDIST[x] for x in self.ACTIVE))) / (n - 2)
        return ans

    def get_avg_dist_from_children(self, i):
        """
        :param i: the node to calculate avg distance of
        :return: the avg distance between node i and its children
        """
        deltaii = self.profile_distance(self.PROFILES[i], self.PROFILES[i])
        normaliser = 1
        children = self.CHILDREN[i]
        for c1 in range(len(children)):
            for c2 in range(c1, len(children)):
                normaliser += 1
                deltaii += self.profile_distance(self.PROFILES[children[c1]], self.PROFILES[children[c2]])
        return deltaii / normaliser

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

    def relaxed_neighbor_joining(self, A):
        """Find the closest node B to A.
        1. Given node A find closest node B.
        2. Given node B find closest node C
        3. IF: Check if C == A then B is closest to A."""

    def neighbor_join_criterion(self, i, j):
        """Get the neighbor join criterion d_u(i,j)-r(i)-r(j) which should be minimized for each join"""
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

    def neighborJoin(self):

        n = len(self.ACTIVE)
        # update the total profile every 200 iterations and at the beginning
        if self.ITERATION % 200 == 0:
            self.update_total_profile()
        self.ITERATION += 1
        # Base case
        if n == 2:
            # Pia: I don't think it makes sense to calculate the weights, up and out distances for the last join.
            # The formulas don't work with n=2 (division by 0) but we also don't need that info anymore after joining
            # everything. I think.
            n1, n2 = self.ACTIVE[0], self.ACTIVE[1]
            self.CHILDREN[(n1,n2)] = [n1,n2]
            self.ACTIVE.remove(n1), self.ACTIVE.remove(n2)
            self.ACTIVE.append((n1, n2))
            return

        # Find min join criterion
        distances = {(i, j): self.neighbor_join_criterion(i, j) for i in self.ACTIVE for j in self.ACTIVE if i != j}
        newNode = min(distances, key=distances.get)
        i, j = newNode
        weight = self.compute_weight(i, j, n)
        self.CHILDREN[newNode] = [i, j]
        self.PROFILES[newNode] = self.merge_profiles(i, j, weight=weight)
        # self.incr_total_profile(i,j,newNode)
        # self.TOTAL_PROFILE -= np.array(self.PROFILES[i]) / n - np.array(self.PROFILES[j]) / n \
        #                       + np.array(self.PROFILES[newNode]) / (n - 1)
        self.UPDIST[newNode] = self.get_updist(i, j, weight)

        self.VARIANCE_CORR[newNode] = weight * self.VARIANCE_CORR[i] + (1 - weight) * self.VARIANCE_CORR[j] \
                                      + weight * (1 - weight) * self.compute_variance(i, j)

        self.ACTIVE.append(newNode)
        self.ACTIVE.remove(i), self.ACTIVE.remove(j)
        self.TOTAL_PROFILE = (np.array(self.TOTAL_PROFILE)*n - np.array(self.PROFILES[i]) - np.array(self.PROFILES[j])
                              + np.array(self.PROFILES[newNode])) / (n - 1)
        return

    def getEdges(self, list, edges):
        i = list[0]
        j = list[1]
        edges.append([i, j])
        if type(i) is str and type(j) is str:
            return edges
        if type(i) is str:
            edges.append(self.getEdges(j, edges))
        elif type(j) is str:
            edges.append(self.getEdges(i,edges))
        else:
            edges.append(self.getEdges(i, edges))
            edges.append(self.getEdges(j, edges))
        return edges

    def makeUnRooted(self):
        n1 = self.ACTIVE[0][0]
        n2 = self.ACTIVE[0][1]
        self.CHILDREN[n2].append(n1)
        self.CHILDREN[n1].append(n2)
        del self.CHILDREN[(n1,n2)]
        newProfile = self.merge_profiles(n1, n2, weight=0.5)
        self.PROFILES[n1] = newProfile
        self.PROFILES[n2] = newProfile
        return

    def recomputeProfiles(self):
        for n in self.PROFILES:
            if n not in self.SEQUENCES:
                profiles = [self.PROFILES[x] for x in self.CHILDREN[n]]
                profile = profiles[0]
                for p in profiles[1:]:
                    profile = [[sum(x) for x in zip(profile[i], p[i])] for i in range(4)]
                self.PROFILES[n] = [[t / len(self.CHILDREN[n]) for t in row] for row in profile]

    def nearestNeighbourInterchange(self):
        end = int(np.log2(len(self.SEQUENCES)) + 1)
        edgesInternal = []
        for i in self.CHILDREN:
            #remove edges coming from root
            if i == self.ACTIVE[0]:
                continue
            for j in self.CHILDREN[i]:
                if j not in self.SEQUENCES:
                    edgesInternal.append([i, j])
        #To have correct profile values for comparing
        self.makeUnRooted()
        print(edgesInternal)
        for _ in range(end):
            for i in edgesInternal:
                A = i[0][0]
                B = i[0][1]
                C = i[1][0]
                D = i[1][1]
                #root of tree
                if i[1] in self.CHILDREN[i[0]] and i[0] not in self.CHILDREN[i[1]]:
                    if i[1] == i[0][0]:
                        A = i[0][1]
                    else:
                        A = i[0][0]
                    for j in self.CHILDREN:
                        if i[0] in self.CHILDREN[j]:
                            B = j
                if i[0] in self.CHILDREN[i[1]] and i[1] not in self.CHILDREN[i[0]]:
                    if i[0] == i[1][0]:
                        C = i[1][1]
                    else:
                        C = i[1][0]
                    for j in self.CHILDREN:
                        if i[1] in self.CHILDREN[j]:
                            D = j

                dABCD = self.corrected_distances(A,B) + self.corrected_distances(C,D)
                dACBD = self.corrected_distances(A,C) + self.corrected_distances(B,D)
                dADBC = self.corrected_distances(A,D) + self.corrected_distances(B,C)

                if dABCD < min(dACBD,dADBC):
                    print("no switch")

                elif dACBD < min(dABCD,dADBC):
                    self.CHILDREN[i[0]].remove(B)
                    self.CHILDREN[i[1]].remove(C)
                    self.CHILDREN[i[0]].append(C)
                    self.CHILDREN[i[1]].append(B)
                    print("switch B, C")

                elif dADBC < min(dABCD, dACBD):
                    self.CHILDREN[i[0]].remove(B)
                    self.CHILDREN[i[1]].remove(D)
                    self.CHILDREN[i[0]].append(D)
                    self.CHILDREN[i[1]].append(B)
                    print("switch B, D")

                self.recomputeProfiles()
        return
