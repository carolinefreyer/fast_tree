import fastTree as ft

# Initialises all the global variables and calls neighbourhood join
def run():
    tree = ft.FastTree()
    tree.initialize_sequences('fasttree-reallysmall.aln')
    tree.initialize_top_hits()
    while(len(tree.ACTIVE)>1):
        tree.neighborJoin()
    
    #print(tree.ACTIVE)
run()