import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from surprise import UnsupervisedSurpriseExtractor


class SurgicalMultiTaskDataset(Dataset):
	# Set use_surprise to False by default as it clashes with top-down surgical domain mapping
	def __init__(self, root_dir=".", split="train", phase_dict=None,
	             patch_size=16, img_size=224, use_surprise=False):
		"""
        Dataset matching directory structure:
          - root_dir/annotations/bbox/by_split/{hands,tools}/{split}.json
          - root_dir/annotations/bbox/by_video/hands/{video_id}/annotations.json
          - root_dir/annotations/bbox/by_video/tool/{video_id}/annotations.json
          - root_dir/images/{video_id}/{frame_filename}
        """
		self.root_dir = root_dir
		self.split = split
		self.patch_size = patch_size
		self.img_size = img_size
		self.grid_size = img_size // patch_size
		self.phase_dict = phase_dict if phase_dict is not None else {}
		self.use_surprise = use_surprise
		
		self.surprise_extractor = UnsupervisedSurpriseExtractor(grid_size=self.grid_size)
		
		# --- AGGRESSIVE DATA AUGMENTATION ---
		# Training gets aggressive augmentations to prevent memorization
		self.train_transform = transforms.Compose([
			transforms.Resize((img_size, img_size)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15),
			transforms.ColorJitter(brightness=0.2, contrast=0.2),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		
		# Validation/Test must remain static and clean
		self.val_transform = transforms.Compose([
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		
		self.transform = self.train_transform if self.split == 'train' else self.val_transform
		
		# Paths to split files
		self.hands_split_path = os.path.join(root_dir, 'annotations', 'bbox', 'by_split', 'hands', f'{split}.json')
		self.tools_split_path = os.path.join(root_dir, 'annotations', 'bbox', 'by_split', 'tools', f'{split}.json')
		
		self.samples = self._build_samples()
	
	def _parse_split_file(self, json_path):
		"""Extracts video IDs or file paths from split JSON."""
		if not os.path.exists(json_path):
			return []
		with open(json_path, 'r') as f:
			data = json.load(f)
		
		if isinstance(data, list):
			return data
		elif isinstance(data, dict):
			if 'videos' in data:
				return data['videos']
			elif 'images' in data:
				return [img.get('file_name', img) if isinstance(img, dict) else img for img in data['images']]
			elif 'files' in data:
				return data['files']
			else:
				return list(data.keys())
		return []
	
	def _load_video_annotations(self, target_folder, video_id):
		"""Loads annotations.json for a given video, properly parsing COCO format."""
		anno_path = os.path.join(self.root_dir, 'annotations', 'bbox', 'by_video', target_folder, str(video_id),
		                         'annotations.json')
		parsed_annos = {}
		
		if os.path.exists(anno_path):
			with open(anno_path, 'r') as f:
				data = json.load(f)
			
			# Check if it's COCO format
			if 'images' in data and 'annotations' in data:
				# 1. Create a quick lookup dictionary: image_id -> file_name
				id_to_name = {img['id']: img['file_name'] for img in data['images']}
				
				# 2. Map every bounding box to its correct file_name
				for anno in data['annotations']:
					img_name = id_to_name.get(anno['image_id'])
					if img_name:
						if img_name not in parsed_annos:
							parsed_annos[img_name] = []
						parsed_annos[img_name].append(anno['bbox'])
			else:
				# Fallback in case some files are already flat dictionaries
				parsed_annos = data
		
		return parsed_annos
	def _validate_split_exclusivity(self, train_files, val_files, test_files):
		"""Ensures that the splits are mutually exclusive at the VIDEO level."""
		
		def get_video_ids(file_list):
			return set([os.path.basename(f).split('_')[0] for f in file_list])
		
		train_vids = get_video_ids(train_files)
		val_vids = get_video_ids(val_files)
		test_vids = get_video_ids(test_files)
		
		train_val_overlap = train_vids & val_vids
		train_test_overlap = train_vids & test_vids
		val_test_overlap = val_vids & test_vids
		
		if train_val_overlap or train_test_overlap or val_test_overlap:
			raise ValueError(f"CRITICAL DATA LEAKAGE DETECTED! Videos are bleeding across splits:\n"
			                 f"Train-Val Overlap: {train_val_overlap}\n"
			                 f"Train-Test Overlap: {train_test_overlap}\n"
			                 f"Val-Test Overlap: {val_test_overlap}")
	
	def _build_samples(self):
		# 1. Gather all valid filenames for this split (from train.json / val.json)
		valid_filenames = set()
		for split_path in [self.hands_split_path, self.tools_split_path]:
			if os.path.exists(split_path):
				with open(split_path, 'r') as f:
					data = json.load(f)
					if isinstance(data, dict) and 'images' in data:
						for img in data['images']:
							# Extract just the filename, ignoring any folders saved in the JSON
							valid_filenames.add(os.path.basename(img['file_name']))
		
		# Validate exclusivity of splits
		if self.split == 'train':
			train_files = valid_filenames
			val_files = set(self._parse_split_file(self.hands_split_path.replace('train', 'val')) +
			                self._parse_split_file(self.tools_split_path.replace('train', 'val')))
			test_files = set(self._parse_split_file(self.hands_split_path.replace('train', 'test')) +
			                 self._parse_split_file(self.tools_split_path.replace('train', 'test')))
			self._validate_split_exclusivity(train_files, val_files, test_files)
			
			print("Split exclusivity validated: No video overlaps found between train, val, and test splits.")
		
		samples = []
		missing_count = 0
		images_dir = os.path.join(self.root_dir, 'images')
		
		if not os.path.exists(images_dir):
			return samples
		
		# 2. Iterate through the actual video folders (01, 02, 03...)
		for vid_folder in sorted(os.listdir(images_dir)):
			vid_path = os.path.join(images_dir, vid_folder)
			if not os.path.isdir(vid_path):
				continue
			
			# 3. Load annotations for this specific video
			hands_annos = self._load_video_annotations('hands', vid_folder)
			tool_annos = self._load_video_annotations('tool', vid_folder)
			
			# 4. Check each image in the folder
			for frame_file in sorted(os.listdir(vid_path)):
				if not frame_file.lower().endswith(('.jpg', '.png', '.jpeg')):
					continue
				
				# 5. ONLY keep the image if it is authorized by the split JSON
				if valid_filenames and frame_file not in valid_filenames:
					continue
				
				rel_path = f"{vid_folder}/{frame_file}"
				
				# --- STRICT INTERSECTION FIX ---
				# Verify the phase label actually exists for this frame before keeping it.
				if self.phase_dict and rel_path not in self.phase_dict and frame_file not in self.phase_dict:
					missing_count += 1
					continue
				
				# 6. Get bounding boxes (checking both with and without the .jpg extension)
				frame_key = os.path.splitext(frame_file)[0]
				h_boxes = hands_annos.get(frame_file, hands_annos.get(frame_key, []))
				t_boxes = tool_annos.get(frame_file, tool_annos.get(frame_key, []))
				
				samples.append({
					'rel_path': rel_path,
					'full_path': os.path.join(vid_path, frame_file),
					'hands_bboxes': h_boxes,
					'tools_bboxes': t_boxes
				})
		
		print(
			f"[{self.split.upper()}] Loaded {len(samples)} valid frames. Dropped {missing_count} ghost frames missing phase labels.")
		return samples
	
	def _create_spatial_grid(self, bboxes, original_w, original_h):
		grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32)
		if not bboxes:
			return grid, False
		
		scale_x = self.img_size / original_w
		scale_y = self.img_size / original_h
		
		for bbox in bboxes:
			if len(bbox) >= 4:
				x_min, y_min, w, h = bbox[:4]
				grid_x_min = int((x_min * scale_x) // self.patch_size)
				grid_y_min = int((y_min * scale_y) // self.patch_size)
				grid_x_max = int(((x_min + w) * scale_x) // self.patch_size)
				grid_y_max = int(((y_min + h) * scale_y) // self.patch_size)
				
				grid_x_min = max(0, min(grid_x_min, self.grid_size - 1))
				grid_y_min = max(0, min(grid_y_min, self.grid_size - 1))
				grid_x_max = max(0, min(grid_x_max, self.grid_size - 1))
				grid_y_max = max(0, min(grid_y_max, self.grid_size - 1))
				
				if grid_x_max >= grid_x_min and grid_y_max >= grid_y_min:
					grid[grid_y_min:grid_y_max + 1, grid_x_min:grid_x_max + 1] = 1.0
		
		return grid, True
	
	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, idx):
		sample = self.samples[idx]
		img_path = sample['full_path']
		rel_path = sample['rel_path']
		
		try:
			image = Image.open(img_path).convert('RGB')
			original_w, original_h = image.size
			image_tensor = self.transform(image)
		except FileNotFoundError:
			image_tensor = torch.zeros((3, self.img_size, self.img_size))
			original_w, original_h = 1920, 1080
		
		# Force a -1 default so missing labels are exposed, not hidden as Phase 0
		phase_id = self.phase_dict.get(rel_path, self.phase_dict.get(os.path.basename(rel_path), -1))
		
		if phase_id == -1:
			raise KeyError(f"CRITICAL: Frame {rel_path} not found in ground truth phase dictionary!")
		
		phase_target = torch.tensor(phase_id, dtype=torch.long)
		
		hands_boxes = sample['hands_bboxes']
		tools_boxes = sample['tools_bboxes']
		all_boxes = hands_boxes + tools_boxes
		
		spatial_grid, has_tools = self._create_spatial_grid(all_boxes, original_w, original_h)
		
		if self.use_surprise:
			surprise_map = self.surprise_extractor.compute_surprise_map(image_tensor)
			spatial_grid = torch.clamp(spatial_grid + 0.5 * surprise_map, max=1.0)
		
		return image_tensor, phase_target, spatial_grid, torch.tensor(has_tools, dtype=torch.bool)