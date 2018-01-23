#!/usr/bin/env bash
echo "Submitting simulation and fit together"
./run_speceff_des_bias_sim.sh && ./run_speceff_des_bias_fit.sh
