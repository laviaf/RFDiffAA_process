# README

## Table of Contents

1. [Prepare for LigandMPNN/ProteinMPNN](#1-prepare-for-ligandmpnn/proteinmpnn)
2. [Set Up Environment](#2-set-up-environment)
3. [Run RFDiffusionAA](#3-run-rfdiffusionaa)
   - [Prepare `config.yaml`](#31-prepare-configyaml)
   - [Run Inference](#32-run-inference)
   - [Filter Generated Backbones](#33-filter-generated-backbones)
4. [Run LigandMPNN](#4-run-ligandmpnn)
   - [Generate Mask Position JSON File](#41-generate-mask-position-json-file)
   - [Run LigandMPNN](#42-run-ligandmpnn)
5. [Run AlphaFold2](#5-run-alphafold2)
   - [Organize Generated Sequences](#51-organize-generated-sequences)
   - [Run AlphaFold2 Predictions](#52-run-alphafold2-predictions)
   - [Filter Generated Structures](#53-filter-generated-structures)

---

## 1. Prepare for LigandMPNN/ProteinMPNN

To run LigandMPNN or ProteinMPNN, install the environment with the following command:

```bash
conda create -n ligandmpnn_env python=3.11
conda activate ligandmpnn_env
conda install pytorch
pip install numpy ProDy ml_collections dm-tree
```

LigandMPNN and ProteinMPNN trained models with a small Gaussian noise (0.02, 0.10, 0.20, 0.30Å) added to the backbone coordinates. In RFDiffusionAA, they used the model trained with 0.20Å noise `proteinmpnn_v_48_020`. The model parameters can be downloaded by

```bash
cd heme_binder_diffusion/lib/LigandMPNN
bash get_model_params.sh
```

LigandMPNN/ProteinMPNN can be run by `python run.py` with predefined parameters. Typically, these parameters are useful:

```bash
--pdb_path # backbone pdb path
--out_folder # output path
--model_type ligand_mpnn (or protein_mpnn)
--checkpoint_ligand_mpnn model_params/ligandmpnn_v_32_020_25.pt
(or --checkpoint_protein_mpnn model_params/proteinmpnn_v_48_020.pt)
# total sequence num = batch_size * number_of_batches
--batch_size 3 
--number_of_batches 5

--fixed_residues "C1 C2 C3 C4 C5 C6 C7 C8 C9 C10" # specify the index of chain and residue whose sequence is fixed
--bias_AA "A:10.0" # increase or decrease the bias towards certain residue types
--omit_AA "ACG" # equivalent to use bias_AA and setting bias to be a large negative number
# set bias for specific residues
--bias_AA_per_residue {"C1": {"G": -0.3, "C": -2.0, "P": 10.8}, "C3": {"P": 10.0}}
--omit_AA_per_residue {"C1": "ACG"}

--ligand_mpnn_use_atom_context 0 # mask all ligand atoms
--ligand_mpnn_use_side_chain_context 1 # use side chain atoms of fixed residues as additional ligand atoms
```

## 2. Set Up Environment

Create the necessary conda environments using the provided YAML files

```bash
conda env create -f heme_binder_diffusion/envs/diffusion.yml
conda env create -f heme_binder_diffusion/envs/mlfold.yml
```

## 3. Run RFDiffusionAA

### 3.1 Prepare config.yaml

Prepare `rf_diffusion_all_atom/config/inference/config.yaml` for RFDiffusionAA.
Typically, you only need to change these parameters:

```bash
### Number of Generated Backbones
inference:
  num_designs: <number_of_designs>
  ligand: <name_of_the_ligand_in_reference_pdb>
### Length of Generated Proteins
contigmap:
  contigs: ["30-110,A64-64,30-110"]
  length: <total_length>
```

"30-110,A64-64,30-110" indicates the 64-th residue is fixed; there are two protein spans before and after this fixed residue, whose lengths are sampled between 30 and 110.
contigmap.length controls the total length of the protein.

### 3.2 Run Inference

Move or copy the customized config.yaml to `<output_dir>`.
Execute the following command in terminal:

```bash
conda activate diffusion
python run_inference.py \
  --config-dir=<output_dir> \
  --config-name=config.yaml \
  inference.input_pdb=<path_to_input_pdb> \
  inference.output_prefix=<output_dir>/0_diffusion \
  inference.design_startnum=0
```

### 3.3 Filter Generated Backbones

Open `rf_diffusion_all_atom/stat.ipynb`, run the section "`Filter RFDiffAA Result`" to calculate the number of residues within 5Å of the selected ligand. Modify this variable in the ipynb file:

```bash
# set path in ipynb file
path=<output_dir>
```

The variables `pos_right` and `pos_left` defined in this section are positions of selected atoms above the DPP in CP_SS_TS. (Positions are selected in pymol using ```iterate_state 1, sele, print(f"{x}, {y}, {z}")```).

Generated backbones with: 1. more than 5 residues within 6Å of selected atoms; and 2. no residue within 3.2Å of ligand is selected. The filtered results are stored in `<output_dir>/0_diffusion_selected`. The statistical result is stored in `<output_dir>/contact_stat.csv`.

## 4. Run ProteinMPNN

Before starting generate sequence, we need to select generated backbones manually. For each selected backbone, copy them to a new folder. For example, if we select the `0_diffusion_71.pdb`.

```
- <new_dir>
	- 0_diffusion
		- 0_diffusion_71.pdb
		- 0_diffusion_71.trb
```

### 4.1 Generate Mask Position JSON File

Open `heme_binder_diffusion/pipeline.ipynb`, modify `WDIR=<new_dir>`. Run the section "1: Running LigandMPNN on diffused backbones" to generate the python command for creating the mask JSON file.
Execute the generated command in the terminal:

```bash
python heme_binder_diffusion/scripts/design/make_maskdict_from_trb.py \
  --out <new_dir>/masked_pos.jsonl \
  --trb <path_to_trb_files>
```

### 4.2 Run ProteinMPNN

`omit_AA` and `bias_AA` can be set as parameters for ProteinMPNN.

Run the ProteinMPNN script in the terminal:

```bash
conda activate diffusion
# design 500 sequences for selected backbone 0_diffusion_71.pdb
python heme_binder_diffusion/lib/LigandMPNN/run.py \
  --model_type ligand_mpnn \
  --fixed_residues_multi <new_dir>/masked_pos.jsonl \
  --out_folder <new_dir>/1_ProteinMPNN \
  --number_of_batches 5 \
  --batch_size 100 \
  --temperature 0.3 \
  --pdb_path <new_dir>/0_diffusion/0_diffusion_71.pdb \
  --checkpoint_ligand_mpnn heme_binder_diffusion/lib/LigandMPNN/model_params/ligandmpnn_v_32_020_25.pt \
  --bias_AA "H:-10.0,C:-10.0,M:-2.0,A:-2.0"
```

Generated sequences will be stored in `<new_dir>/1_ProteinMPNN`.

## 5. Run AlphaFold2

### 5.1 Organize Generated Sequences

Open `heme_binder_diffusion/pipeline.ipynb`, run the section "2: Running AlphaFold2" to prepare the command for running AlphaFold2.

### 5.2 Run AlphaFold2 Predictions

```bash
conda activate mlfold
cd <new_dir>/2_af2
bash commands_af2
```

The AlphaFold2 predicted structure will be stored in `<new_dir>/2_af2`.

### 5.3 Filter Generated Structures

Open `rf_diffusion_all_atom/stat.ipynb`, run the section `Analyze AF2 Output`.
In the subsection `Align AF2`, we align AlphaFold2-generated structures with RFDiffusionAA-designed backbones.
In the subsection `Align ligand with fix residue`, we align the ligand with the selected residue (e.g., H61). The `fix_res_id` should be modified.
In the subsection `Stat rmsd, plddt, clash`, we calculate RMSD, pLDDT scores, and number of residues within 3.2Å of the ligand.

## 6. Redesign Pocket with LigandMPNN

