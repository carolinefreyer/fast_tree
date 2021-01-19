import fastTree as ft


# Initialises all the global variables and calls neighbourhood join
def run():
    output_file = open('../data/fasttree-newick-output.txt', 'w')
    tree = ft.FastTree()
    tree.initialize_sequences('../data/fasttree-reallysmall.aln')
    tree.initialize_top_hits()
    while (len(tree.ACTIVE) > 1):
        tree.neighbor_join()
    tree.nearest_neighbour_interchange()
    tree.compute_branch_lenghts()
    newickFormat = tree.newick_format(tree.ACTIVE[0], ";")
    newickFormat += ";"
    output_file.write(newickFormat)
    print(newickFormat)


run()
