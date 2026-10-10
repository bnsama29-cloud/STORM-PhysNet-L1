# Leakage audit (record)

`AUDIT_FINDINGS.md` documents the problems found in the original pipeline: back-filled gaps, centred despiking, Kp look-ahead, windows crossing split boundaries, and a mismatch between the paper and the code. `verify_storm_physnet.py` checks those findings against the original code at commit `72008c7`. It is kept as a record. The fixes live in `rerun_2026_10/kit/storm_fixed/data.py`.
