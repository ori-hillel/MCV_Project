import torch
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image


def main():
	target_file = "paper_gaze_outputs/test_student_gaze.pt"
	print(f"Loading gaze data from {target_file}...")
	
	try:
		payload = torch.load(target_file, map_location="cpu")
		samples = payload["samples"]
	except FileNotFoundError:
		print(f"Error: Could not find {target_file}. Did you run generate_test_gaze.py?")
		return
	
	print(f"Successfully loaded {len(samples)} frames. Selecting 5 random samples...")
	
	# Pick 5 random frames to visualize
	indices = random.sample(range(len(samples)), min(5, len(samples)))
	
	fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
	if len(indices) == 1:
		axes = [axes]
	
	for ax, idx in zip(axes, indices):
		sample = samples[idx]
		img_path = sample["path"]
		
		# The 14x14 probability grid
		gaze_probs = sample["gaze_probs"].float()
		
		# Load the original image
		try:
			img = Image.open(img_path).convert("RGB")
		except FileNotFoundError:
			ax.set_title("Image Not Found")
			ax.axis("off")
			continue
		
		w, h = img.size
		
		# Upsample the 14x14 grid to match the original image resolution
		heatmap = F.interpolate(
			gaze_probs.unsqueeze(0).unsqueeze(0),
			size=(h, w),
			mode="bilinear",
			align_corners=False
		).squeeze().numpy()
		
		# Find the cell with the highest probability (the "gaze point")
		flat_idx = int(gaze_probs.argmax().item())
		gw = gaze_probs.shape[1]
		
		gy = flat_idx // gw
		gx = flat_idx % gw
		
		# Convert 14x14 grid coordinates to actual image pixel coordinates
		pred_x = (gx + 0.5) / gw * w
		pred_y = (gy + 0.5) / gw * h
		
		# Plotting
		ax.imshow(img)
		ax.imshow(heatmap, alpha=0.45, cmap="jet")
		ax.scatter(pred_x, pred_y, marker="x", s=150, linewidths=3, c="red")
		
		# Clean up the filename for the title
		filename = img_path.replace("\\", "/").split("/")[-1]
		ax.set_title(filename)
		ax.axis("off")
	
	plt.tight_layout()
	plt.suptitle("ViT Student Gaze Predictions (Red X = Max Probability)", y=1.05, fontsize=16)
	plt.show()


if __name__ == "__main__":
	main()