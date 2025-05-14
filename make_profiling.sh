#!/bin/bash

metrics=$( nvprof --query-metrics | grep -e "hit_rate" -e "stall" -e "occupancy" -e "eligible_warps" -e "ipc" | awk '{print $1}' | tr ':\n' ',' | sed "s/,,/,/g" | sed "s/,$//" );
dataset="datasets/viscousFingering.vtu";

# BUILD
MAIN="src/alg/critPoints.cu" RUNTIME_ARGS="--export /dev/null" ./test.sh > profiling_configure.out 2>&1
if [[ $? -ne 0 ]]; then
    echo "Build or test failed; exiting...";
    exit $?;
fi;

gpu=$( nvidia-smi --query-gpu name --format=csv | tail -n 1 | sed "s/NVIDIA[ \t]*//g" | sed "s/ /_/g" );
program="build_${HOSTNAME}/./main --input ${dataset} --export /dev/null";
# RUN GENERAL TIMING
todo="nvprof ${program} 2>profiling_${gpu}.out";
echo "${todo}";
eval "${todo}";
rval=$?;
if [[ $rval -ne 0 ]]; then
    echo "General Profiling failed; exiting...";
    exit $rval;
fi;
echo "END_GENERAL_PROFILING" >>profiling_${gpu}.out;
# RUN METRIC TIMING
todo="nvprof --metrics ${metrics} --csv ${program} 2>>profiling_${gpu}.out";
echo "${todo}";
eval "${todo}";
rval=$?;
if [[ $rval -ne 0 ]]; then
    echo "Metric Profiling failed; exiting...";
    exit $rval;
fi;
echo "END_METRIC_PROFILING" >>profiling_${gpu}.out;
# POSTPROCESS
python3 profiling_postprocess.py profiling_${gpu}.out report_${gpu}.out;

