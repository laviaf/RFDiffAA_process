DATA_DIR=$1
TOTAL_VAL=$2
# TARGET_PREFIX=$2

# trb=""
# for ((i=0;i<$TOTAL_VAL;i++)); do
#     trb=${trb}" "${DATA_DIR}/0_diffusion/0_diffusion_${i}.trb 
# done

for ((i=0;i<$TOTAL_VAL;i++)); do
    python heme_binder_diffusion/lib/LigandMPNN/run.py \
        --model_type protein_mpnn \
        --ligand_mpnn_use_atom_context 0 \
        --fixed_residues_multi ${DATA_DIR}/masked_pos.jsonl \
        --out_folder ${DATA_DIR}/1_ProteinMPNN \
        --number_of_batches 100 \
        --temperature 0.3 \
        --pdb_path ${DATA_DIR}/0_diffusion/0_diffusion_${i}.pdb \
        --checkpoint_protein_mpnn heme_binder_diffusion/lib/LigandMPNN/model_params/proteinmpnn_v_48_020.pt \
        --bias_AA "H:-10.0,C:-10.0,M:-2.0,A:-2.0"
done