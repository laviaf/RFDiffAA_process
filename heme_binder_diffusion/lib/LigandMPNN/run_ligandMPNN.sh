DATA_PATH=$1

for ((i=0; i<100; i++));do
    python heme_binder_diffusion/lib/LigandMPNN/run.py \
        --model_type ligand_mpnn \
        --fixed_residues_multi ${DATA_PATH}/masked_pos.jsonl \
        --out_folder ${DATA_PATH}/1_LigandMPNN \
        --number_of_batches 8 \
        --temperature 0.3 \
        --omit_AA CM \
        --pdb_path ${DATA_PATH}/0_diffusion/0_diffusion_${i}.pdb \
        --checkpoint_ligand_mpnn heme_binder_diffusion/lib/LigandMPNN/model_params/ligandmpnn_v_32_020_25.pt
done