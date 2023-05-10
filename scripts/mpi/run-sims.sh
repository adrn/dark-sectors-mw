#!/bin/bash
#SBATCH -J stream-sims
#SBATCH -o logs/sims.o
#SBATCH -e logs/sims.e
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p cca
# -C 'rome&ib'

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/dark-sectors-mw/scripts

init_env

date

# python run_sims.py --dist=20 --overwrite --overwriteplots

# mpirun python -m mpi4py.futures run_sims.py --dist=10 --mpi --overwriteplots
mpirun python -m mpi4py.futures run_sims.py --dist=20 --mpi --grid=gallery
# mpirun python -m mpi4py.futures run_sims.py --dist=30 --mpi --overwriteplots

date
