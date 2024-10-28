# README

## Table of Contents

1. [Set Up Environment](#1-set-up-environment)
2. [Run RFDiffusionAA](#2-run-rfdiffusionaa)
   - [Prepare `config.yaml`](#21-prepare-configyaml)
   - [Run Inference](#22-run-inference)
   - [Filter Generated Backbones](#23-filter-generated-backbones)
3. [Run LigandMPNN](#3-run-ligandmpnn)
   - [Generate Mask Position JSON File](#31-generate-mask-position-json-file)
   - [Run LigandMPNN](#32-run-ligandmpnn)
4. [Run AlphaFold2](#4-run-alphafold2)
   - [Organize Generated Sequences](#41-organize-generated-sequences)
   - [Run AlphaFold2 Predictions](#42-run-alphafold2-predictions)
   - [Filter Generated Structures](#43-filter-generated-structures)

---

## 1. Set Up Environment

Create the necessary conda environments using the provided YAML files:

```bash
conda env create -f heme_binder_diffusion/envs/diffusion.yml
conda env create -f heme_binder_diffusion/envs/mlfold.yml
```

## 2. Run RFDiffusionAA

### 2.1 Prepare config.yaml

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

### 2.2 Run Inference

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

### 2.3 Filter Generated Backbones

Open `rf_diffusion_all_atom/stat.ipynb`, run the section "`Filter RFDiffAA Result`" to calculate the number of residues within 5Å of the selected ligand. Modify this variable in the ipynb file:

```bash
# set path in ipynb file
path=<output_dir>
```

The variables `pos_right` and `pos_left` defined in this section are positions of selected atoms above the DPP in CP_SS_TS. (Positions are selected in pymol using ```iterate_state 1, sele, print(f"{x}, {y}, {z}")```).

Generated backbones with: 1. more than 5 residues within 6Å of selected atoms; and 2. no residue within 3.2Å of ligand is selected. The filtered results are stored in `<output_dir>/0_diffusion_selected`. The statistical result is stored in `<output_dir>/contact_stat.csv`.

## 3. Run LigandMPNN

Before starting generate sequence, we need to select generated backbones manually. For each selected backbone, copy them to a new folder. For example, if we select the `0_diffusion_71.pdb`.

```
- <new_dir>
	- 0_diffusion
		- 0_diffusion_71.pdb
		- 0_diffusion_71.trb
```

### 3.1 Generate Mask Position JSON File

Open `heme_binder_diffusion/pipeline.ipynb`, modify `WDIR=<new_dir>`. Run the section "1: Running LigandMPNN on diffused backbones" to generate the python command for creating the mask JSON file.
Execute the generated command in the terminal:

```bash
python heme_binder_diffusion/scripts/design/make_maskdict_from_trb.py \
  --out <new_dir>/masked_pos.jsonl \
  --trb <path_to_trb_files>
```

### 3.2 Run LigandMPNN

In `heme_binder_diffusion/lib/LigandMPNN/run_ligandMPNN.sh`, `omit_AA` and `bias_AA` can be set as parameters for LigandMPNN.

Run the LigandMPNN script in the terminal:

```bash
conda activate diffusion
bash heme_binder_diffusion/lib/LigandMPNN/run_ligandMPNN.sh <new_dir> # provide the absolute path
```

Generated sequences will be stored in `<new_dir>/1_LigandMPNN`.

## 4. Run AlphaFold2

### 4.1 Organize Generated Sequences

Open `heme_binder_diffusion/pipeline.ipynb`, run the section "2: Running AlphaFold2" to prepare the command for running AlphaFold2.

### 4.2 Run AlphaFold2 Predictions

```bash
conda activate mlfold
cd <new_dir>/2_af2
bash commands_af2
```

The AlphaFold2 predicted structure will be stored in `<new_dir>/2_af2`.

### 4.3 Filter Generated Structures

Open `rf_diffusion_all_atom/stat.ipynb`, run the section `Analyze AF2 Output`.
In the subsection `Align AF2`, we align AlphaFold2-generated structures with RFDiffusionAA-designed backbones.
In the subsection `Align ligand with fix residue`, we align the ligand with the selected residue (e.g., H61). The `fix_res_id` should be modified.
In the subsection `Stat rmsd, plddt, clash', we calculate RMSD, pLDDT scores, and number of residues within 3.2Å of the ligand.

