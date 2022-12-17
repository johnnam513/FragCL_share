#!/usr/bin/env bash

source $HOME/.bashrc
#conda activate GraphMVP


echo $@
date
echo "start"
python pretrain_fragcl3d_angles.py $@
echo "end"
date
