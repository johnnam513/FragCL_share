#!/bin/bash

#cd ../src_classification

mode_list=(
#GraphFrag_randomaug_double/GEOM_2D_nmol50000_cut_anybond/epochs_100_0_alin9
#GraphFrag_randomaug_double/GEOM_2D_nmol50000_cut_CCsinglebond/epochs_100_0_alin9
#GraphFrag_randomaug_fp/GEOM_2D_nmol50000_cut_singlebond/epochs_100_0_neg_0.7_thres_0.7_b1024

#GraphFrag_randomaug/GEOM_2D_nmol50000_cut_singlebond/epochs_100_0.0_aug_0.3_edgepert
fragcl3d_4frag_mae/GEOM_2D_3D_nmol50000_cut_singlebond/epochs_100_0.2_aug_0.1_normalize_3daug_1.0_1000epoch_app
#rebuttal_mgssl
)

export gnn='gin'
for mode in "${mode_list[@]}"; do
    echo "$mode"
    ls ../output/"$mode"
    ls
    bash run_fine_tuning_model_mae.sh "$mode"

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