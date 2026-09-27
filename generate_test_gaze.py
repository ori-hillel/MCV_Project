import os
import torch
from PIL import Image
from tqdm import tqdm

# Import the architecture and dataset utilities your friends wrote
from gaze_student import ViTGazeStudent, gaze_distribution
from gaze_data import collect_split_frames, flatten_video_dict, student_image_transform


def load_weights_safely(model, weights_path, device):
	"""Safely loads weights whether they are saved as a state_dict or a full checkpoint."""
	checkpoint = torch.load(weights_path, map_location=device)
	if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
		model.load_state_dict(checkpoint['state_dict'])
	elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
		model.load_state_dict(checkpoint['model_state_dict'])
	else:
		model.load_state_dict(checkpoint)
	return model


def main():
	# Hardcoded configurations for IDE 'Run' button
	root_dir = "."
	split = "test"
	weights_path = "paper_gaze_outputs/best_gaze_student.pt"
	output_path = "paper_gaze_outputs/test_student_gaze.pt"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	print(f"Loading ViTGazeStudent from {weights_path}...")
	model = ViTGazeStudent(dropout=0.0).to(device)
	model = load_weights_safely(model, weights_path, device)
	model.eval()
	
	print(f"Collecting frames for {split} split...")
	videos = collect_split_frames(root_dir, split)
	frame_paths = flatten_video_dict(videos)
	
	if not frame_paths:
		raise ValueError(f"CRITICAL: No frames found for the {split} split!")
	
	samples = []
	print(f"Generating 14x14 gaze grids for {len(frame_paths)} frames...")
	
	with torch.no_grad():
		for path in tqdm(frame_paths, desc="Inferring Gaze"):
			# Load and transform image using the student transform
			with Image.open(path) as img:
				img_rgb = img.convert("RGB")
				img_tensor = student_image_transform(img_rgb).unsqueeze(0).to(device)
			
			# Forward pass
			logits = model(img_tensor)
			probs = gaze_distribution(logits)  # Converts logits to spatial probabilities
			
			# Store results (converting to float16 to save disk space)
			samples.append({
				"path": path,
				"gaze_probs": probs.squeeze().cpu().to(torch.float16)
			})
	
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	
	# Save payload matching the formatting style of the pseudo gaze generator
	payload = {
		"metadata": {
			"root_dir": str(os.path.abspath(root_dir)),
			"split": split,
			"source": "vit_student_inference"
		},
		"samples": samples
	}
	
	torch.save(payload, output_path)
	print(f"Success! Saved test gaze distributions to {output_path}")


if __name__ == "__main__":
	main()