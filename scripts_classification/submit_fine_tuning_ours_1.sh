#!/bin/bash

#cd ../src_classification

mode_list=(
#GraphFrag_randomaug_double/GEOM_2D_nmol50000_cut_anybond/epochs_100_0_alin9
#GraphFrag_randomaug_double/GEOM_2D_nmol50000_cut_CCsinglebond/epochs_100_0_alin9
#GraphFrag_randomaug_fp/GEOM_2D_nmol50000_cut_singlebond/epochs_100_0_neg_0.7_thres_0.7_b1024

#GraphFrag_randomaug/GEOM_2D_nmol50000_cut_singlebond/epochs_100_0.0_aug_0.3_edgepert
#fragcl3d_4frag/GEOM_2D_3D_nmol50000_cut_singlebond/epochs_100_0.2_aug_0.1_normalize_1daug_1.0
#fragcl3d_dihedral/GEOM_2D_3D_nmol50000_cut_singlebond_manual_0911_dihedral_from_frag/epochs_100_0.2_aug_0.1_0912_onlydihedral_v4
#fragcl3d_dihedral/GEOM_2D_3D_nmol50000_cut_singlebond_manual_0911_dihedral_from_frag/epochs_100_0.0_aug_0.1_1105_spherenet_nodihed
#rebuttal_mgssl
#FragCL_4frags/GEOM_2D_nmol50000_4frag_cut_singlebond/epochs_100_0.0_choosetwo_0.1_2
BricsCL/GEOM_2D_nmol50000_brics_with_dummy/epochs_100_0.0_choosetwo_0.1_4
)

export gnn='gin'
for mode in "${mode_list[@]}"; do
    echo "$mode"
    ls ../output/"$mode"
    ls
    bash run_fine_tuning_model_1.sh "$mode"

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
