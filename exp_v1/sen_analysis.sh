#!/bin/bash
num_samples=(20 25)
hazards=("SLE" "MCE")
criteria=("Probability" "Energy")
for num_sample in "${num_samples[@]}"; do
  printf "\033[1;32mNumber of samples:\t%s\n\033[0m" "$num_sample"
  for hazard in "${hazards[@]}"; do
    printf "\033[1;32mHazard:\t%s\n\033[0m" "$hazard"
    for criterion in "${criteria[@]}"; do
      printf "\033[1;32mCriterion:\t%s\n\033[0m" "$criterion"
      if [ "$hazard" == "SLE" ]; then
        python3 sen_analysis.py --num_samples "$num_sample" --hazard "$hazard" --criteria "$criterion"
      else
        python3 sen_analysis.py --num_samples "$num_sample" --hazard "$hazard" --criteria "$criterion" \
                                --xi 0.0001 --factor 4000 --rot 0.05
      fi
    done
  done
done
