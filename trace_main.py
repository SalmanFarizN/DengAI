# trace_main.py
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath("src"))

# Import your actual entry point
from main import main  # assuming you have a main() function in src/main.py

graphviz = GraphvizOutput()
graphviz.output_file = 'callgraph.png'

with PyCallGraph(output=graphviz):
    main()
