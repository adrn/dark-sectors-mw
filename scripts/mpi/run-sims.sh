#!/bin/bash
#SBATCH -J stream-sims
#SBATCH -o logs/sims.o
#SBATCH -e logs/sims.e
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p gen
#SBATCH -C 'rome&ib'

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/dark-sectors-mw/scripts

init_env

date

mpirun python -m mpi4py.futures run_sims.py --dist=10 --mpi
# python run_sims.py --dist=20
# python run_sims.py --dist=30

date
