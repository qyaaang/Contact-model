#!/bin/bash
hazards=("SLE" "MCE")
for hazard in "${hazards[@]}"; do
    printf "\033[1;32mHazard:\t%s\n\033[0m" "$hazard"
    if [ "$hazard" == "SLE" ]; then
        python3 sen_analysis.py  --hazard "$hazard"
    else
      python3 sen_analysis.py --hazard "$hazard" --criteria --xi 0.0001 --factor 4000 --rot 0.05
    fi
done
