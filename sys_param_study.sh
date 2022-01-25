#!/bin/bash
hazards=("SLE" "MCE")
mbs=(1 5 10 15 20 25)
con_param=(1. 1000. 1000. 0.1)
for hazard in "${hazards[@]}"; do
  printf "\033[1;32mHazard:\t%s\n\033[0m" "$hazard"
  if [ "$hazard" == "SLE" ]; then
    for mb in "${mbs[@]}"; do
      python3 param_study.py --mb "$mb" --hazard "$hazard" --g "${con_param[0]}" --r_k "${con_param[1]}" \
                             --r_c "${con_param[2]}" --mu "${con_param[3]}"
    done
  else
    for mb in "${mbs[@]}"; do
      python3 param_study.py --mb "$mb" --hazard "$hazard" --g "${con_param[0]}" --r_k "${con_param[1]}" \
                             --r_c "${con_param[2]}" --mu "${con_param[3]}" --xi 0.0001 --factor 4000 \
                             --rot 0.05
    done
  fi
done
