#!/usr/bin/env bash

source $HOME/.bashrc
#conda activate GraphMVP


echo $@
date
echo "start"
python pretrain_fragcl3d_dihedral_united.py $@
echo "end"
date
