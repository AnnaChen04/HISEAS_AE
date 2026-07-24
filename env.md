# Conda 100 for *BenPo*🐻‍❄️!
**Prof. Woag 🐢**

## What is conda and where is conda
Conda is a package manager that helps to compartmentalize running environment for independent projects, on 🐻‍❄️💻, it is at

> /opt/miniconda3/envs 

The particular python interpreter, i.e., the actual python "machine" lives under the directory `/opt/miniconda3/envs/env_name/bin`

For example, for an env named `lig_sst`, the python interpretor would be at:
> /opt/miniconda3/envs/lig_sst/bin/python3.10


## How to create an env 

Create an .yml file that contains the names of libraries you need in thsi environment, i.e., the libaries you will use in the project. 

Then type the following command in terminal:

> conda env create -n "myenv"

## How to actiavet an env 

> conda activate "myenv"

## How to download a new library into conda env
For instance to download `scipy`:
>conda install -c conda-forge scipy

## To delete a package/library
The inverse of the above opeartion : 
> conda remove scipy

## Duplicate Environment 
Say, you want to either 
- Change the name of your environment 
- Have a new environment that builds upon a current environemnt( a superset)
1. Suppose you have a environment.yml file, then just copy it to the new repo and do 

> conda create -n "new_name" environment.yml

2. Suppose you don't have the reciepie envrionemnt.yml file, then frist generate it by:

> conda export ..

## **The most Important!**
To check all command available in conda:

> conda -h 

To check the syntax for particular command, e.g., env

> conda en -h

## PS how to print emoji(For very Ben BenPo)
1. command+shift+space 
2. fn + E