#!/usr/bin/env bash

cd "$(dirname "$0")"

file="$DES3YR/config/FITOPTS_NO_SYST.DAT"
base="$DES3YR/base/DES_BASE_FIT.nml"
version="$DES3YR/config/SIMGEN_DES_AM_BIAS.input"

../scripts/submit_fit.sh $file $base $version
