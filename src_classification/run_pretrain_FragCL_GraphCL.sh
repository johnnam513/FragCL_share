#!/usr/bin/env bash

source $HOME/.bashrc
#conda activate GraphMVP

echo $@
date

echo "start"
python pretrain_FragCL_GraphCL.py $@
echo "end"
date
