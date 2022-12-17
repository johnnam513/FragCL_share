#!/usr/bin/env bash

cd ../src_classification

#export dataset=GEOM_2D_nmol50000_nconf5_nupper1000
export dataset=GEOM_2D_3D_nmol50000_singlebond_manual_1207_sorted_dihedral_from_frag_nohydanchor
export dropout_ratio=0.0
export epochs=100
export aug_strength=0.1
export aug_mode=choosetwo

export time=3
#export mode_list=(EP IG AM CP GraphLoG Motif Contextual GraphCL JOAO JOAOv2)
export mode_list=(EP IG AM CP GraphLoG Motif Contextual JOAO JOAOv2)

#export time=12
#export mode_list=(GPT_GNN)
export mode_list=(FragCL_GraphCL)

for mode in "${mode_list[@]}"; do
     export folder="$mode"/"$dataset"/epochs_"$epochs"_"$dropout_ratio"_"$aug_mode"_"$aug_strength"_4
     echo "$folder"

     mkdir -p ../output/"$folder"

     export output_file=../output/"$folder"/pretraining.out
     export output_model_dir=../output/"$folder"/pretraining
     
     
     if [[ ! -f "$output_file" ]]; then
          echo "$folder" undone

          #sbatch --gres=gpu:v100l:1 -c 8 --mem=32G -t "$time":00:00  --account=rrg-bengioy-ad --qos=high --job-name=baselines \
          #--output="$output_file" \
          bash run_pretrain_"$mode".sh \
          --epochs="$epochs" \
          --dataset="$dataset" \
          --batch_size=256 \
          --dropout_ratio="$dropout_ratio" --num_workers=8 \
          --output_model_dir="$output_model_dir" \
	  --aug_mode="choosetwo" \
	  --choose=0 \
	  --gnn_type='gin'\
	  --aug_strength="$aug_strength"
     fi
done
