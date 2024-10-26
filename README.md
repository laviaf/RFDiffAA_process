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

Create the necessary Conda environments using the provided YAML files:

```bash
conda env create -f heme_binder_diffusion/envs/diffusion.yml
conda env create -f heme_binder_diffusion/envs/mlfold.yml
```

## 2. Run RFDiffusionAA
### 2.1 Prepare config.yaml
Refer to rf_diffusion_all_atom/config/inference/config.yaml for the configuration template.
Customize the following parameters in config.yaml:

```bash
### Number of Generated Backbones
inference:
  num_designs: <number_of_designs>
### Length of Generated Proteins
contigmap:
  contigs: ["30-110,A64-64,30-110"]
  length: <total_length>
```

"30-110,A64-64,30-110": The 64th residue is fixed; the lengths before and after are sampled between 30 and 110 residues.
length: Total length of the protein.
Move or copy the customized config.yaml to your output directory <output_dir>.

### 2.2 Run Inference
Execute the RFDiffusionAA inference:

```bash
python run_inference.py \
  --config-dir=<output_dir> \
  --config-name=config.yaml \
  inference.input_pdb=<path_to_input_pdb> \
  inference.output_prefix=<output_dir>/0_diffusion \
  inference.design_startnum=0
```

### 2.3 Filter Generated Backbones
In rf_diffusion_all_atom/stat.ipynb, run the section "Stat 5A residue num" to calculate the number of residues within 5 Å of the selected ligand part. For CP_SS_TS, select ligand atoms above the DPP.

## 3. Run LigandMPNN
### 3.1 Generate Mask Position JSON File
In heme_binder_diffusion/pipeline.ipynb, run section "1: Running LigandMPNN on diffused backbones" to generate the command for creating the mask JSON file.
Execute the generated command in the terminal:
```bash
python heme_binder_diffusion/scripts/design/make_maskdict_from_trb.py \
  --out <output_dir>/masked_pos.jsonl \
  --trb <path_to_trb_files>
```

### 3.2 Run LigandMPNN
Run the LigandMPNN script:
``` bash
bash heme_binder_diffusion/lib/LigandMPNN/run_ligandMPNN.sh <output_dir>
```

## 4. Run AlphaFold2
### 4.1 Organize Generated Sequences
In heme_binder_diffusion/pipeline.ipynb, run "2: Running AlphaFold2" to prepare the sequences for AlphaFold2.

### 4.2 Run AlphaFold2 Predictions
```bash 
cd <path_to_input_pdb>/2_af2
bash commands_af2
```

### 4.3 Filter Generated Structures
In rf_diffusion_all_atom/stat.ipynb, run
```bash
"Align AF2": Align AlphaFold2-generated structures with RFDiffusionAA-designed backbones.
"Align ligand with fix residue": Align the ligand with the selected residue (e.g., H61).
"Stat rmsd, plddt, clash": Calculate RMSD, pLDDT scores, and check for structural clashes.
```
