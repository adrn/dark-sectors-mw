#!/bin/bash
#SBATCH -J stream-sims
#SBATCH -o logs/sims.o
#SBATCH -e logs/sims.e
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -p cca
#SBATCH -C rome,ib

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/dark-sectors-mw/scripts

init_env

date

python run_sims.py --overwrite

date
