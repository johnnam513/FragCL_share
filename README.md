# FragCL codebase

## Environments
Install packages under conda env
```bash
conda create -n GraphMVP python=3.7
conda activate GraphMVP

conda install -y -c rdkit rdkit
conda install -y -c pytorch pytorch=1.9.1
conda install -y numpy networkx scikit-learn
pip install ase
pip install git+https://github.com/bp-kelley/descriptastorus
pip install ogb
export TORCH=1.9.0
export CUDA=cu102  # cu102, cu110

wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric==1.7.2
```

Or just use

```
conda env create -f fragcl.yaml
```


## Dataset Preprocessing
For data preprocessing with brics decomposition, please use the following commands (please fill this up jaehyun):
```
cd src_classification
python GEOM_dataset_preparation_brics_with_dummy.py --data_folder ../datasets --n_mol 50000
cd ..
```

Then, the processed dataset will be saved at **datasets/molecule_datasets/GEOM_2D_nmol50000_brics_with_dummy/**.

If you want to run the ICLR-submitted version (cutting a single bond in half), use the following commands:
```
cd src_classification
python GEOM_dataset_preparation_allsingle_2d.py --data_folder ../datasets --n_mol 50000
cd ..
```

Then, the processed dataset will be saved at **datasets/molecule_datasets/GEOM_2D_nmol50000_frag_cut_singlebond/**.

## Pretraining
For pretraining, please use the followng commands:
```
cd scripts_classification
bash submit_pretraining_fragcl2d.sh
cd ..
```

## Fine-tuning
Before fine-tuning, please copy the fine-tuning datasets by following commands:
```
cd datasets
mkdir molecule_datasets
scp -r alin@143.248.158.138:/home/jaehyun/GraphMVP/datasets/molecule_datasets/bace/ datasets
scp -r alin@143.248.158.138:/home/jaehyun/GraphMVP/datasets/molecule_datasets/bbbp/ datasets
scp -r alin@143.248.158.138:/home/jaehyun/GraphMVP/datasets/molecule_datasets/sider/ datasets
scp -r alin@143.248.158.138:/home/jaehyun/GraphMVP/datasets/molecule_datasets/tox21/ datasets
scp -r alin@143.248.158.138:/home/jaehyun/GraphMVP/datasets/molecule_datasets/toxcast/ datasets
scp -r alin@143.248.158.138:/home/jaehyun/GraphMVP/datasets/molecule_datasets/clintox/ datasets
scp -r alin@143.248.158.138:/home/jaehyun/GraphMVP/datasets/molecule_datasets/muv/ datasets
scp -r alin@143.248.158.138:/home/jaehyun/GraphMVP/datasets/molecule_datasets/hiv/ datasets
cd ..
```

For fine-tuning for random seeds \{0,1,2\}, please use the following commands:
```
cd scripts_classification
bash submit_fine_tuning_fragcl2d.sh
bash submit_fine_tuning_fragcl2d_1.sh
bash submit_fine_tuning_fragcl2d_2.sh
cd ..
```

Then, the model will be saved at **output/GraphFrag_randomaug_weighted_neg** folder.

For jaehyun: To update github, please use the following commands after you change files in **jaehyun@143.248.158.138:~/GraphMVP** (Note: **datasets** folder is ignored by .gitignore) :
```
git add .
git commit -m "what you want to comment"
git push origin main
```
