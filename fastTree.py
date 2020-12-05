import numpy as np

# Disimilarity matrix between the alphabet A,C,G,T
ACGT_DIS = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]


class FastTree(object):
    def __init__(self):
        self.SEQUENCES = {}
        self.PROFILES = {}
        self.CHILDREN = {}
        self.UPDIST = {}
        self.ACTIVE = []
        self.NODENUM = 0

    def initialize_sequences(self, filename):
        """
        Returns a 4 by L matrix with the frequency of each nulceotide at each position

        Input:
              @seq: String of nucleotides from the alphabet A,C,G,T
        """
        f = open(filename, 'r')
        lines = f.readlines()
        for i in range(int(len(lines) / 2)):
            tmp1 = lines[2 * i].strip()[1:]
            tmp2 = lines[2 * i + 1].strip()
            self.SEQUENCES[tmp2] = tmp1
            # @ Caroline: I swapped the keys and values of the sequences dict, as it is easier (the actual
            # sequence is used much more often than its name)
        for i in self.SEQUENCES:
            self.CHILDREN[i] = []
            self.UPDIST[i] = 0
            self.ACTIVE.append(i)

    def initialize_profiles(self):
        for seq in self.SEQUENCES:
            freq = np.zeros((4, len(seq)))
            for i in range(len(seq)):
                if seq[i] == 'A':
                    freq[0][i] = 1
                elif seq[i] == 'C':
                    freq[1][i] = 1
                elif seq[i] == 'G':
                    freq[2][i] = 1
                elif seq[i] == 'T':
                    freq[3][i] = 1
            self.PROFILES[seq] = freq

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

    def out_distance(self, profile, T, i, n):
        """
        Calculates the out-distance for node i
        r(i) = (n*delta(profile,T) - delta(i,i) - (n-1)*upDist(i)-sum_{k =\=i} upDist(k))/(n-2)
        Input: @profile: profile of node i
               @T: total profile
               @i: node number
               @n: number of active nodes
        """
        deltaii = 0
        normaliser = 0
        for c1 in range(len(self.CHILDREN[i])):
            for c2 in range(c1, len(self.CHILDREN[i])):
                normaliser += 1
                deltaii += self.profile_distance(self.PROFILES[c1], self.PROFILES[c2])
        if normaliser != 0:
            deltaii = deltaii / normaliser
        return (n * self.profile_distance(profile, T) - deltaii - (n - 2) * self.UPDIST[i]
                - sum(list(self.UPDIST.values()))) / (n - 2)

    def merge_profiles(self, seqs):
        """
        Calculates the profile of multiple sequences by "merging" their profiles.
        This method can be used to calculate the total profile by inputting all sequences.
        Input: @seqs: the sequences whose profiles will be merged

        """
        L = len(seqs)
        profiles = [self.PROFILES[x] for x in seqs]
        T = profiles[0]
        for p in profiles[1:]:
            T = [[sum(x) for x in zip(T[i], p[i])] for i in range(4)]
        T = [[t / L for t in row] for row in T]
        return T

    def neighborJoin(self):
        if len(self.ACTIVE) == 2:
            n1, n2 = self.ACTIVE[0], self.ACTIVE[1]
            dist = self.uncorrected_distance(n1,n2)
            self.CHILDREN[(n1, n2)] = [n1, n2]
            self.UPDIST[(n1, n2)] = dist / 2
            # Add to tree & return tree - not sure about the data structure for this
        distances = {(i, j): self.uncorrected_distance(i, j) for i in self.ACTIVE for j in self.ACTIVE if i != j}
        newNode = min(distances, key=distances.get)
        i, j = newNode
        self.CHILDREN[newNode] = [i, j]
        self.PROFILES[newNode] = self.merge_profiles([i, j])
        self.ACTIVE.append(newNode)
        self.ACTIVE.remove(i), self.ACTIVE.remove(j)
        self.UPDIST[newNode] = distances[min(distances, key=distances.get)] / 2  # UNWEIGHTED JOINS!

        # Compute total profile T

        # Find min join criterion
        # Join = uncorrectedDistance(i,j) - r(i) - r(j)
        # let min_i, min_j be the nodes to join
        # create a new node with new UPDIST, CHILDREN, PROFILE values
        # remove min_i and min_j from active list & add new node
        # recursive call
        # add nodes to tree
        # return tree
