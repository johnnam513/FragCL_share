#!/bin/bash

#cd ../src_classification

mode_list=(
#random
#EP/GEOM_2D_nmol50000_nconf5_nupper1000/epochs_100_0
#AM/GEOM_2D_nmol50000_nconf5_nupper1000/epochs_100_0
#IG/GEOM_2D_nmol50000_nconf5_nupper1000/epochs_100_0
#CP/GEOM_2D_nmol50000_nconf5_nupper1000/epochs_100_0
#GraphFrag/GEOM_2D_nmol50000/epochs_100_0
#GraphLoG/GEOM_2D_nmol50000_nconf5_nupper1000/epochs_100_0
#Motif/GEOM_2D_nmol50000_nconf5_nupper1000/epochs_100_0
#Contextual/GEOM_2D_nmol50000_nconf5_nupper1000/epochs_100_0
GraphCL/GEOM_2D_nmol50000_nconf5_nupper1000/epochs_100_0
#JOAO/GEOM_2D_nmol50000_nconf5_nupper1000/epochs_100_0
#JOAOv2/GEOM_2D_nmol50000_nconf5_nupper1000/epochs_100_0

#GraphMVP/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_VAE_1/6_51_10_0.1/0.15_EBM_dot_prod_0.2_normalize_l2_detach_target_2_100_0
#GraphMVP_hybrid/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_VAE_1_AM_1/6_51_10_0.1/0.15_EBM_dot_prod_0.05_normalize_l2_detach_target_2_100_0
#GraphMVP_hybrid/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_VAE_1_CP_0.1/6_51_10_0.1/0.15_EBM_dot_prod_0.2_normalize_l2_detach_target_2_100_0
)
for mode in "${mode_list[@]}"; do
    echo "$mode"
    ls ../output/"$mode"
    ls
    bash run_fine_tuning_model_2.sh "$mode"

    echo

done



# Below is for ablation study
mode_list=(
GraphMVP/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_VAE_1/6_51_10_0.1/0.15_EBM_dot_prod_0.2_normalize_l2_detach_target_2_100_0
GraphMVP/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_VAE_1/6_51_10_0.1/0.15_InfoNCE_dot_prod_0.2_normalize_l2_detach_target_2_100_0
GraphMVP/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_VAE_0/6_51_10_0.1/0.15_InfoNCE_dot_prod_0.2_normalize_l2_detach_target_2_100_0
GraphMVP/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_VAE_0/6_51_10_0.1/0.15_EBM_dot_prod_0.2_normalize_l2_detach_target_2_100_0
GraphMVP/GEOM_3D_nmol50000_nconf5_nupper1000/CL_0_VAE_1/6_51_10_0.1/0.15_EBM_dot_prod_0.2_normalize_l2_detach_target_2_100_0
GraphMVP/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_AE_1/6_51_10_0.1/0.15_InfoNCE_dot_prod_0.2_normalize_l2_detach_target_2_100_0
GraphMVP/GEOM_3D_nmol50000_nconf5_nupper1000/CL_1_AE_1/6_51_10_0.1/0.15_EBM_dot_prod_0.2_normalize_l2_detach_target_2_100_0
GraphMVP/GEOM_3D_nmol50000_nconf5_nupper1000/CL_0_AE_1/6_51_10_0.1/0.15_EBM_dot_prod_0.2_normalize_l2_detach_target_2_100_0
)
