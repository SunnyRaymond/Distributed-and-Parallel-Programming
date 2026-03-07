# DAS-5 Quick Usage Guide

This document records the basic workflow for logging into the **DAS-5 cluster** and requesting a **GPU node (TitanRTX)** for computation.

---

# 1. Login to DAS-5

Connect to the front-end filesystem node:

```bash
ssh dpp2567@fs2.das5.science.uva.nl
```

After login you should see a prompt similar to:

```
[dpp2567@fs2 ~]$
```

---

# 2. Navigate to Large Scratch Disk

The home directory has limited quota. Use the **scratch disk** for projects and datasets.

```bash
cd /var/scratch/dpp2567
```

Confirm location:

```bash
pwd
```

Expected:

```
/var/scratch/dpp2567
```

---

# 3. Request a GPU Node (TitanRTX)

Request an interactive GPU session:

```bash
srun -p fatq --gres=gpu:1 --constraint=TitanRTX --pty bash -l
```

Explanation:

| Option                  | Meaning                                |
| ----------------------- | -------------------------------------- |
| `-p fatq`               | use GPU queue                          |
| `--gres=gpu:1`          | request 1 GPU                          |
| `--constraint=TitanRTX` | ensure Titan RTX GPU                   |
| `--pty bash`            | open interactive shell on compute node |

After allocation the prompt changes, for example:

```
[dpp2567@node221 ~]$
```

---

# 4. Load CUDA Toolkit

Load CUDA module required for GPU computation:

```bash
module load cuda12.6/toolkit/12.6
```

Check available modules if needed:

```bash
module avail cuda
```

---

# 5. Verify GPU

Check whether the GPU is visible:

```bash
nvidia-smi
```

Typical output shows:

* GPU model
* memory usage
* running processes

Example:

```
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        |
| 0    Titan RTX               |
+-----------------------------------------------------------------------------+
```

---

# 6. Enable Environment

Initialize conda:

```bash
source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh
```

Activate an environment:

```bash
conda activate <env_name>
```

Example:

```bash
conda activate cs224n_dfp
```

Change uv cache dir:

```bash
export XDG_CACHE_HOME=/var/scratch/dpp2567/.cache
```

---

# 7. Useful Commands

Check job queue:

```bash
squeue -u dpp2567
```

Check GPU nodes:

```bash
sinfo -p fatq
```

Exit compute node:

```bash
exit
```

---

# Typical Workflow

```bash
ssh dpp2567@fs2.das5.science.uva.nl
cd /var/scratch/dpp2567
srun -p fatq --gres=gpu:1 --constraint=TitanRTX --pty bash
module load cuda12.6/toolkit/12.6
nvidia-smi
source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh
conda activate my_env
```

---

