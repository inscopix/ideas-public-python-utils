![coverage](coverage.svg)

# IDEAS Python Utils

This repository contains utilities and functions to help you
build tools for IDEAS. 


## Installation

You do not have to install this to use (unless you are developing this).
If so, see [below](#developing)


## Usage


### In a project using [Pipfile](https://github.com/pypa/pipfile)

If you're building a tool in IDEAS, you are probably working 
on a project that uses a `Pipfile`. Include this repo in your IDEAS tool
by adding this to your  `Pipfile`:

```bash
[packages]
ideas-python-utils = {ref = "main", git = "https://${IDEAS_GITHUB_TOKEN}@github.com/inscopix/ideas-python-utils.git"}
```


> You can omit the `IDEAS_GITHUB_TOKEN` when this repo becomes public. 




## Developing 


### Prerequisites

- python
- make
- git
- You should have SSH keys set up with github

If you are working on developing this, download and install using:


```bash
git clone git@github.com:inscopix/ideas-python-utils.git
cd ideas-python-utils
poetry install  # this is a "editable" install
```

If you use Jupyter Lab, and you want this kernel available to your
global install of Jupyter Lab, use this:

```bash
make jupyter
```

This will create a kernel called `ideas_python_utils` that you can access
from Jupyter (Lab or notebook). 

## Run tests 

### Locally

```bash
make test
```


### On remote. 

Tests should run via Github Actions on push/merge to `main`. This is automatic. 


## License 