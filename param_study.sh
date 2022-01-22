#!/bin/bash
param="sys"
hazards=("SLE")
if [ "$param" == "sys" ]; then
  mbs=(5 10 15 20 25)
  con_param=(1. 1000. 1000. 0.1)
  for hazard in "${hazards[@]}"; do
    printf "\033[1;32mHazard:\t%s\n\033[0m" "$hazard"
    for mb in "${mbs[@]}"; do
      python3 param_study.py --mb "$mb" --g "${con_param[0]}" --r_k "${con_param[1]}" --r_c "${con_param[2]}" \
                             --mu "${con_param[3]}"
    done
  done
else
  pass
fi
