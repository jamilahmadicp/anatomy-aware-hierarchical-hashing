# Manifest schema

Required: `path, split, anatomy_label, fine_label`.

Recommended: `sample_id, patient_id, study_id, abnormal_label, joint_label, irma_code`.

Allowed split values used by the scripts: `train`, `val`, `test`.

For IRMA, the package intentionally requires the user to provide the manuscript's exact mapping from fine-grained class to one of the 16 anatomy categories. It does not infer this mapping from filenames or external assumptions.

For MURA, `prepare_mura_manifest.py` parses the standard folder layout. It carves a patient-disjoint internal validation subset from the official training partition and reserves the official validation partition as `test`.
