#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 100GB
#SBATCH --mail-type NONE
#SBATCH --time 1:00:00

# Add --ntasks <#CPUS> --gpus <model>:1 flags to sbatch of this script

# Configuration
override=0; # Set to 1 for override
n_repeats=3; # Times to repeat each dataset

cd /home/tlranda/TetraTopo_GPU/GALE_Vanilla;
source /home/tlranda/TetraTopo_GPU/env.sh; # Modulefiles and list

# Set number of CPUs once
ncpus=$( lscpu | grep -e "^CPU(s):" | awk '{print $NF}' );
# GALE should use #(logical cores minus 2) as threads for best performance
full_subscribe=$((${ncpus}-2));
echo "Execute on host ${HOSTNAME}";
nvidia-smi;
echo "NCPUS: ${ncpus}";
echo "Full subscribe: ${full_subscribe}";

outputdir="${HOSTNAME}_outputs";
mkdir -p ${outputdir};

binary="ttkScalarFieldCriticalPoints";

datasets=( datasets/1k_preprocess/Bucket_1k.vtu
           datasets/1k_preprocess/Engine_1k.vtu
           datasets/1k_preprocess/viscousFingering_1k.vtu
           datasets/1k_preprocess/Foot_1k.vtu
           datasets/1k_preprocess/Fish_1k.vtu
           datasets/1k_preprocess/Asteroid_1k.vtu
           datasets/1k_preprocess/Hole_1k.vtu
           datasets/1k_preprocess/ctBones_1k.vtu
           datasets/1k_preprocess/Stent_1k.vtu);
declare -A target_array=( [datasets/1k_preprocess/Bucket_1k.vtu]=Result
                          [datasets/1k_preprocess/Engine_1k.vtu]=Scalars_
                          [datasets/1k_preprocess/viscousFingering_1k.vtu]=concentration
                          [datasets/1k_preprocess/Foot_1k.vtu]=Scalars_
                          [datasets/1k_preprocess/Fish_1k.vtu]=Elevation
                          [datasets/1k_preprocess/Asteroid_1k.vtu]=scalar
                          [datasets/1k_preprocess/Hole_1k.vtu]=Result    # -b 32
                          [datasets/1k_preprocess/ctBones_1k.vtu]=Scalars_
                          [datasets/1k_preprocess/Stent_1k.vtu]=Scalars_
                         );
for ds in ${datasets[@]}; do
    echo "${ds}";
    shortname=$( basename ${ds} | tr "." "\n" | head -n 1 );
    # Determine the dataset's proper array to target
    used_array="${target_array[$ds]}";
    if [[ ! ${used_array+_} ]]; then
        # Auto-lookup for attached arrays, filtered down to hopefully just the PointData segment excluding any array named "_index"
        found_arrays=$( ./build_n01_2026/bin/ttkScalarFieldCriticalPoints -i ${ds} -l | sed -n '/.*- \w/,/.*CellData:/p' | awk '{print $NF}' | grep -v -e "CellData:" -e "_index" | sort | uniq );
        # Default to picking the first array from the list
        used_array=$( echo ${found_arrays} | tr " " "\n" | head -n 1 );
        echo "No array set for ${ds}; found arrays: ${found_arrays}";
    fi;
    if [[ "${used_array}" == "" ]]; then
        echo -e "\t! No array; skipping !";
        continue;
    fi;
    # Array selected; proceed
    echo -e "\tUsing array: ${used_array}";
    cmd="./build_n01_2026/bin/${binary} -i ${ds} -a ${used_array} -t ${full_subscribe}";
    for iteration in `seq 1 ${n_repeats}`; do
        to_make="${outputdir}/${shortname}_${full_subscribe}CPUS_iter_${iteration}.output";
	if [[ ! -e "${to_make}" || ${override} == 1 ]]; then
	    full_command="${cmd} > ${to_make}";
	    echo "${full_command}";
            eval "${full_command}";
	else
	    echo "${to_make} exists, skipping...";
	fi;
    done;
done;
# Analyze all captured traces
python3 processGALE.py ${outputdir}/*.output;

