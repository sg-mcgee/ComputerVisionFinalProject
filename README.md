The cvproject.py file can run only when dolly_zoom_v3.npy is present in the file. This numpy matrix is produced with zoom_homography_finding.py.

This project utilizes CV2 to mimic a dolly zoom using computational means, rather than camera lens effects. By applying a homographic warp to the frame,
the perspective can be shifted to mirror the effect from AppleTV's "Severance", the inspiration for the project.
