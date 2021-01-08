import fastTree as ft


# Initialises all the global variables and calls neighbourhood join
def run():
    tree = ft.FastTree()
    tree.initialize_sequences('fasttree-reallysmall.aln')
    while (len(tree.ACTIVE) > 1):
        tree.neighborJoin()
    print(tree.ACTIVE)
    tree.nearestNeighbourInterchange()
    tree.recLength(tree.CHILDREN[tree.ACTIVE[0]])
    tree.BRANCH_LENGTHS = {_: round(x, 3) for _, x in tree.BRANCH_LENGTHS.items()}
    newickFormat = tree.newickFormat(tree.ACTIVE[0], "")
    newickFormat += ";"
    print(newickFormat)


run()
