#!/bin/bash
param_names=("Gap" "Stiffness" "Damping" "Friction")
hazards=("SLE" "MCE")
for param_name in "${param_names[@]}"; do
  printf "\033[1;32mParameter name:\t%s\n\033[0m" "$param_name"
  for hazard in "${hazards[@]}"; do
    printf "\033[1;32mHazard:\t%s\n\033[0m" "$hazard"
    if [ "$hazard" == "SLE" ]; then
      if [ "$param_name" == "Gap" ]; then
        python3 con_param_study.py --hazard "$hazard" --param_name "$param_name" --g_lb 0.5 --g_rb 2.0
      else
        python3 con_param_study.py --hazard "$hazard" --param_name "$param_name"
      fi
    else
      if [ "$param_name" == "Gap" ]; then
        python3 con_param_study.py --hazard "$hazard" --param_name "$param_name" --g_lb 0.5 --g_rb 5.0 \
                                   --xi 0.0001 --rot 0.05 --factor 4000
      else
        python3 con_param_study.py --hazard "$hazard" --param_name "$param_name" --xi 0.0001 --rot 0.05 --factor 4000
      fi
    fi
  done
done
