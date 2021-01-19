import fastTree as ft

# Initialises all the global variables and calls neighbourhood join
def run():
    tree = ft.FastTree()
    tree.initialize_sequences('data/fasttree-reallysmall.aln')
    tree.initialize_top_hits()
    while(len(tree.ACTIVE)>1):
        tree.neighborJoin()
    tree.computeBranchLenghts()
    print(tree.newickFormat(tree.ACTIVE[0],""))
    tree.nearestNeighbourInterchange()
    tree.computeBranchLenghts()
    newickFormat = tree.newickFormat(tree.ACTIVE[0],"")
    newickFormat += ";"
    print(newickFormat)

run()