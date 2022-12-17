#!/usr/bin/env bash

source $HOME/.bashrc
#conda activate GraphMVP


echo $@
date
echo "start"
python pretrain_FragCL_transformer3.py $@
echo "end"
date
