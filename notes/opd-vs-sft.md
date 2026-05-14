# OPD vs SFT

SFT trains on fixed demonstrations. The student usually does not generate the training trajectory that is being supervised.

OPD trains on states visited by the student. The teacher or feedback source supervises those student-generated states, which reduces exposure bias from training only on teacher trajectories.

## Classification rule

- SFT on teacher-generated traces: `not_opd`.
- SFT cold start followed by OPD: classify the OPD stage separately.
- SFT cold start followed by reward-only RLVR: `adjacent`, not strict OPD.

