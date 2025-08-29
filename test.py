import matplotlib.pyplot as plt
import torch

from diffdrr.drr import DRR

#from diffdrr.data import load_example_mr

from diffdrr.data import load_example_mammo
from diffdrr.visualization import plot_drr

# Read in the volume and get its origin and spacing in world coordinates
#subject = load_example_mr()
subject = load_example_mammo()

'''The pixel dimensions of the rendered DRR = height × width (e.g., 200 × 200 px).
The physical extent of the detector in world space = pixel count × pixel spacing.
In your case: 200 × 2.0 mm = 400 mm.
So the combination of height and delx controls both the resolution (px) and the scaling in world units (mm),
which determines how big the anatomy appears when projected.'''

# Initialize the DRR module for generating synthetic X-rays
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drr = DRR(
    subject,     # An object storing the CT volume, origin, and voxel spacing
    sdd=1020.0,  # Source-to-detector distance (i.e., focal length)
    height=560,  # Image height (if width is not provided, the generated DRR is square)
    #height=200,  # Image height (if width is not provided, the generated DRR is square)
    delx=2.0,    # Pixel spacing (in mm)
).to(device)

# Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)
rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)

# 📸 Also note that DiffDRR can take many representations of SO(3) 📸
# For example, quaternions, rotation matrix, axis-angle, etc...
img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
plot_drr(img, ticks=False)
plt.show()