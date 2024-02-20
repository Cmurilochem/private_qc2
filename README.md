# This is a private version of the [public qc2](https://github.com/qc2nl/qc2.git) repo

### Remote URLs
To check our remote URL, do:
```console
git remote -v                                                    
```
We are going to see the following addresses:
```console
origin	https://github.com/Cmurilochem/private_qc2.git (fetch)
origin	https://github.com/Cmurilochem/private_qc2.git (push)
public-fork	https://github.com/Cmurilochem/qc2.git (fetch)
public-fork	https://github.com/Cmurilochem/qc2.git (push)
```

### Pushing to the private repo
If you want to push and contribute/create a PR, on the private repo, please, do so via `origin`:
```console
git push origin <branch>
```
Or if you want to pull/fetch/merge using any private branch:
```console
git pull origin <branch>
```

### Updating the private repo with the public fork
```console
git pull public-fork <branch>
```

# Quantum Computing for Quantum Chemistry
<!-- (Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.) -->
[![Build Status](https://github.com/qc2nl/qc2/actions/workflows/build.yml/badge.svg)](https://github.com/qc2nl/qc2/actions)
[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](git@github.com:qc2nl/qc2)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![RSD](https://img.shields.io/badge/rsd-qc2-00a3e3.svg)](https://www.research-software.nl/software/qc2)
[![workflow pypi badge](https://img.shields.io/pypi/v/qc2.svg?colorB=blue)](https://pypi.python.org/project/qc2/)



<!-- | fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](git@github.com:qc2nl/qc2) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/qc2nl/qc2)](git@github.com:qc2nl/qc2) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-qc2-00a3e3.svg)](https://www.research-software.nl/software/qc2) [![workflow pypi badge](https://img.shields.io/pypi/v/qc2.svg?colorB=blue)](https://pypi.python.org/project/qc2/) |
| (4/5) citation                     | [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>) |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Static analysis                    | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=qc2nl_qc2&metric=alert_status)](https://sonarcloud.io/dashboard?id=qc2nl_qc2) |
| Coverage                           | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=qc2nl_qc2&metric=coverage)](https://sonarcloud.io/dashboard?id=qc2nl_qc2) |
| Documentation                      | [![Documentation Status](https://readthedocs.org/projects/qc2/badge/?version=latest)](https://qc2.readthedocs.io/en/latest/?badge=latest) |
| **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](git@github.com:qc2nl/qc2/actions/workflows/build.yml/badge.svg)](git@github.com:qc2nl/qc2/actions/workflows/build.yml) |
| Citation data consistency               | [![cffconvert](git@github.com:qc2nl/qc2/actions/workflows/cffconvert.yml/badge.svg)](git@github.com:qc2nl/qc2/actions/workflows/cffconvert.yml) |
| SonarCloud                         | [![sonarcloud](git@github.com:qc2nl/qc2/actions/workflows/sonarcloud.yml/badge.svg)](git@github.com:qc2nl/qc2/actions/workflows/sonarcloud.yml) |
| MarkDown link checker              | [![markdown-link-check](git@github.com:qc2nl/qc2/actions/workflows/markdown-link-check.yml/badge.svg)](git@github.com:qc2nl/qc2/actions/workflows/markdown-link-check.yml) | -->


## Installation

To install qc2 from GitHub repository, do:

```console
git clone git@github.com:qc2nl/qc2.git
cd qc2
python3 -m pip install -e .
```

In this current version, qc2 can perform `VQE` calculations using both `Qiskit` and `Pennylane`. However, the latter is an optional dependency. To install `Pennylane` and perform automatic testing with it, follow these steps:
```console
git clone git@github.com:qc2nl/qc2.git
cd qc2
python3 -m pip install -e .[pennylane] # (use ".[pennylane]" if you have zsh shell)
```
