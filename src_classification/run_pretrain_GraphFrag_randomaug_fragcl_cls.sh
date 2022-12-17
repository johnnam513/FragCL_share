#!/usr/bin/env bash

source $HOME/.bashrc
#conda activate GraphMVP

echo $@
date

echo "start"
python pretrain_GraphFrag_randomaug_fragcl_cls.py $@
echo "end"
date
