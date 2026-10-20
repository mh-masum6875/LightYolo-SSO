# Continuity task
The original binary (normal/anomaly) task is NOT regenerated here to avoid duplicating 31,067 images.
It is the existing curated dataset at ../../../GitHub/Dataset/Dataset/ with the frozen split
recorded in rebuild-v2/data/splits/ (WP0, checksummed). Use data/wsn_dataset.yaml from the
LightYolo-SSO repo pointed at that folder. The threshold rule scores perfectly on this task by
construction (circular labels); the paper states this openly (Plain-Language Guide 2.1, 5).
