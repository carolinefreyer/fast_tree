import fastTree as ft

# Initialises all the global variables and calls neighbourhood join
def run():
    tree = ft.FastTree()
    tree.initialize_sequences('fasttree-reallysmall.aln')
    tree.initialize_top_hits()
    while(len(tree.ACTIVE)>1):
        tree.neighborJoin()
    newickFormat = tree.newickFormat(tree.ACTIVE[0], "")
    newickFormat += ";"
    print(newickFormat)

run()