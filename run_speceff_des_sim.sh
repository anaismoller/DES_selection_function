#!/usr/bin/env bash
cd "$(dirname "$0")"
file="$DES3YR/config/SIMGEN_DES_SPECEFF.input"
../scripts/submit_sim.sh $file


