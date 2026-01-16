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

mkdir -p datasets/8kep_preprocess;

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
declare -A n_points=( [Bucket]=9958
                      [Engine]=697216
                      [viscousFingering]=1048576
                      [Foot]=1818576
                      [Fish]=2217304
                      [Asteroid]=4184453
                      [Hole]=4634596
                      [ctBones]=8388608
                      [Stent]=8678234
                    );

cd build/bin;

for data in ${!to_make[@]}; do
    if [[ -e ../../datasets/8kep_preprocess/${data}_8kep.vtu ]]; then
        echo "8kep Partition for ${data} exists. Skipping.";
        continue;
    fi;
    echo "Make 8kep dataset for ${data} using ${n_points[${data}]} points";
    if [[ ! -e ../../datasets/8kep_preprocess/${data}_8k.vtu ]]; then
        # 1) Strip for re-indexing
        python3 export_specific.py ../../datasets/base/${data}.vtu --keep ${to_make[${data}]} --export stripped.vtu;
        # 2) Re-index with timing
        ./ttkCompactTriangulationPreconditioningCmd -i stripped.vtu -o ${data}_8k -a ${to_make[${data}]} -b ${n_points[${data}]} | grep "Complete";
        mv ${data}_8k_port_0.vtu ../../datasets/8kep_preprocess/${data}_8k.vtu;
    fi;
    # 3) Add external arrays
    python3 ../../../TetraGPU/hand/export_specific.py ../../datasets/8kep_preprocess/${data}_8k.vtu --export ../../datasets/8kep_preprocess/${data}_8kep.vtu --no-drops --add-external | grep "Non-IO time";
    # 4) Verify (optional)
    #python3 check_size.py ../../datasets/8kep_preprocess/${data}_8kep.vtu;
    date;
done;


