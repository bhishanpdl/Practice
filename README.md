Using Sphinx and Read The Docs
===============================

Basic repo demonstrating usage of Sphinx documentation and
ReadTheDocs integration.

#### Requires:
Sphinx 1.5.6

```bash
python -c """import sphinx; print(sphinx.__version__)"""
```

#### Usage
Use the following instructions to automatically build HTML documentation from the repo.
Suppose we have a package **pkg** and a main package script
`pkg/pkg.py` and two supporting scripts `foo1.py` and `bar1.py`.  

We want to build these scripts.

First we create a initializer script `pkg/__init__.py` with
contents

```py
from .bar1 import bar1
from .foo1 import foo1
from .pkg import Pkg
from .pkg import foo
```

```bash
sphinx-quickstart -q -p "Bhishan's" -a "Bhishan Poudel" -v 1 -r 1 \
 --ext-autodoc --ext-doctest --ext-viewcode --ext-imgmath \
 --no-batchfile --sep docs


 
```
