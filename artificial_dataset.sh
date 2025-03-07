#!/bin/bash

FORCE=1;

if ! [[ -e "mwe.vtu" ]] || [[ ${FORCE} -eq 1 ]]; then
    python3 make_mesh.py --n-points 5 --output mwe;
else
    echo "mwe exists";
fi
if ! [[ -e "tiny.vtu" ]] || [[ ${FORCE} -eq 1 ]]; then
    python3 make_mesh.py --n-points 100 --output tiny;
else
    echo "tiny exists";
fi
if ! [[ -e "big.vtu" ]] || [[ ${FORCE} -eq 1 ]]; then
    python3 make_mesh.py --n-points 100000 --output big;
else
    echo "big exists";
fi

