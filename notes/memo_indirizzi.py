import os

# da usare in /workspace
script_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(script_path)


# esempio cartella /workspace/gradino1/gradino2
script_path = os.path.abspath(__file__) 

# cartella /workspace/gradino1
script_dir = os.path.dirname(script_path)

# /workspace
cartella_padre = os.path.dirname(script_dir)

# setto /workspace come root
PROJECT_ROOT = os.path.dirname(cartella_padre)

