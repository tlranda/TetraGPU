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

cd /home/tlranda/TetraTopo_GPU/TetraGPU;
source ../env.sh; # Modulefiles and list

# Set number of CPUs once
ncpus=$( lscpu | grep -e "^CPU(s):" | awk '{print $NF}' );
full_subscribe=1; # No CPU parallelism, really
echo "Execute on host ${HOSTNAME}";
nvidia-smi;
ngpus=$( nvidia-smi -L | wc -l );
echo "NCPUS: ${ncpus} | NGPUS: ${ngpus}";
echo "Full subscribe: ${full_subscribe}";

mkdir -p ${HOSTNAME}_outputs;

#datasets=$( ls -d datasets/*.vtu );
#datasets=( datasets/Bucket_8.vtu
#           datasets/Engine_8.vtu
#           datasets/viscousFingering_8.vtu
#           datasets/Foot_8.vtu
#          datasets/Fish_8.vtu
#          datasets/Asteroid_8.vtu
#          datasets/Hole_8.vtu
#          datasets/ctBones_8pt.vtu
#          datasets/Stent_8pt.vtu);
#declare -A target_array=( [datasets/Bucket_8.vtu]=Result
#                         [datasets/Engine_8.vtu]=Scalars_
#                         [datasets/viscousFingering_8.vtu]=concentration
#                         [datasets/Foot_8.vtu]=Scalars_
#                         [datasets/Fish_8.vtu]=Elevation
#                         [datasets/Asteroid_8.vtu]=scalar
#                         [datasets/Hole_8.vtu]=Result
#                         [datasets/ctBones_8pt.vtu]=Scalars_
#                         [datasets/Stent_8pt.vtu]=Scalars_
#                        );
#declare -A memory_limits=( [datasets/Bucket_8.vtu]=N
#                          [datasets/Engine_8.vtu]=N
#                          [datasets/viscousFingering_8.vtu]=N
#                          [datasets/Foot_8.vtu]=N
#                          [datasets/Fish_8.vtu]=N
#                          [datasets/Asteroid_8.vtu]=N
#                          [datasets/Hole_8.vtu]=N #151
#                          [datasets/ctBones_8pt.vtu]=N #78
#                           [datasets/Stent_8pt.vtu]=N #73
#                          );
datasets=( hand/ctBones_8ep.vtu );
declare -A target_array=( [hand/ctBones_8ep.vtu]=Scalars_ );
declare -A memory_limits=( [hand/ctBones_8ep.vtu]=N );
#datasets=( datasets/ctBones_8pt.vtu );
#declare -A target_array=( [datasets/ctBones_8pt.vtu]=Scalars_ );
#declare -A memory_limits=( [datasets/ctBones_8pt.vtu]=N );
for ds in ${datasets[@]}; do
    echo "${ds}";
    shortname=$( basename ${ds} | tr "." "\n" | head -n 1 );
    # Determine the dataset's proper array to target
    used_array="${target_array[$ds]}";
    if [[ ! ${used_array+_} ]]; then
        # Auto-lookup for attached arrays, filtered down to hopefully just the PointData segment excluding any array named "_index"
        found_arrays=$( ./build/bin/ttkScalarFieldCriticalPoints -i ${ds} -l | sed -n '/.*- \w/,/.*CellData:/p' | awk '{print $NF}' | grep -v -e "CellData:" -e "_index" | sort | uniq );
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
    cmd="build_${HOSTNAME}/./main --input ${ds} --arrayname ${used_array} -p _index -d 0 -t ${full_subscribe} --export /dev/null";
    limit_size="${memory_limits[$ds]}";
    if [[ "${limit_size}" != "N" ]]; then
        cmd="${cmd} --max_VV ${limit_size}";
        echo -e "\tLimiting max adjacency to ${limit_size}";
    fi;
    for gpu_count in `seq 1 $ngpus`; do
        base_cmd="${cmd} -g ${gpu_count}";
        for iteration in `seq 1 ${n_repeats}`; do
            to_make="${HOSTNAME}_outputs/${shortname}_${gpu_count}_GPUS_${full_subscribe}CPUS_iter_${iteration}.output";
            if [[ ! -e "${to_make}" || ${override} == 1 ]]; then
                full_command="${base_cmd} > ${to_make}";
                echo "${full_command}";
                cmd_rval=$(eval "${full_command}");
                if [[ ${cmd_rval} -ne 0 ]]; then
                    echo "BAD EXIT CODE: ${cmd_rval}";
                    break;
                fi;
            else
                echo "${to_make} exists, skipping...";
            fi;
        done;
        # Analyze all captured traces
        python3 processours.py ${HOSTNAME}_outputs/${shortname}_${gpu_count}_GPUS*.output;
    done;
done;

