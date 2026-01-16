#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 100GB
#SBATCH --mail-type NONE
#SBATCH --time 1:00:00

# Add --ntasks <#CPUS> --gpus <model>:1 flags to sbatch of this script

cd "$(dirname "$0")";
pwd;
source ../env.sh; # Modulefiles and list
# Add ACTOPO libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/tlranda/TetraTopo_GPU/ACTOPO/install/lib64";

lscpu;
nvidia-smi;
date;

mkdir -p datasets/1k_preprocess;

declare -A to_make=(  [Bucket]=Result
                      [Engine]=Scalars_
                      [viscousFingering]=concentration
                      [Foot]=Scalars_
                      [Fish]=Elevation
                      [Asteroid]=scalar
                      [Hole]=Result
                      [ctBones]=Scalars_
                      [Stent]=Scalars_
                   );

cd build/bin;

for data in ${!to_make[@]}; do
    if [[ -e ../../datasets/1k_preprocess/${data}_1k.vtu ]]; then
        echo "1k Partition for ${data} exists. Skipping.";
        continue;
    fi;
    echo "Make 1k dataset for ${data}";
    # 1) Strip for re-indexing
    python3 export_specific.py ../../datasets/base/${data}.vtu --keep ${to_make[${data}]} --export stripped.vtu;
    # 2) Re-index with timing
    ./ttkCompactTriangulationPreconditioningCmd -i stripped.vtu -o ${data}_1k -a ${to_make[${data}]} -b 1000 | grep "Complete";
    mv ${data}_1k_port_0.vtu ../../datasets/1k_preprocess/${data}_1k.vtu;
    # 3) Verify (optional)
    #python3 check_size.py ../../datasets/1k_preprocess/${data}_1k.vtu;
    date;
done;

