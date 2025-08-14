# 이 스크립트는 Python 3.11+ 환경에서 실행해주세요
import argparse
import json
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Moondream2CaptionGenerator:
    def __init__(self, data_dir, emotic_pre_dir, model_revision="2025-06-21", device="cuda", caption_length="short"):
        """
        Initialize caption generator using Moondream2 model
        Requires Python 3.10+
        """
        self.data_dir = data_dir
        self.emotic_pre_dir = emotic_pre_dir
        self.device = device
        self.model_revision = model_revision
        self.caption_length = caption_length
        
        # Initialize model variables
        self.model = None
        self.tokenizer = None
        
        # Load existing CSV files
        self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing CSV files to get image metadata"""
        self.datasets = {}
        
        for split in ['train', 'val', 'test']:
            csv_path = os.path.join(self.emotic_pre_dir, f'{split}.csv')
            if os.path.exists(csv_path):
                logger.info(f"Loading {split} CSV file...")
                df = pd.read_csv(csv_path)
                self.datasets[split] = df
                logger.info(f"Loaded {len(df)} samples from {split} split")
            else:
                logger.warning(f"CSV file not found: {csv_path}")
        
        if not self.datasets:
            raise FileNotFoundError("No CSV files found in the preprocessed directory")
    
    def load_model(self):
        """Load Moondream2 model"""
        if self.model is not None:
            return
            
        logger.info("Loading Moondream2 model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision=self.model_revision,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map={"": self.device} if torch.cuda.is_available() else "auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "vikhyatk/moondream2", 
                revision=self.model_revision
            )
            logger.info("Moondream2 model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_caption(self, image_path):
        """Generate caption for a single image using Moondream2"""
        try:
            if self.model is None:
                self.load_model()
                
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Generate caption using Moondream2's caption method
            caption_result = self.model.caption(
                image,
                length=self.caption_length  # Use instance setting
            )
            
            return caption_result["caption"]
            
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return None
    
    def get_unique_images_from_csv(self, split):
        """Get unique images from a specific split's CSV file"""
        if split not in self.datasets:
            return {}
            
        df = self.datasets[split]
        unique_images = {}
        
        for _, row in df.iterrows():
            filename = row['Filename']
            folder = row['Folder']
            
            if filename not in unique_images:
                # Try different possible paths
                possible_paths = [
                    os.path.join(self.data_dir, 'emotic', folder, filename),
                    os.path.join(self.data_dir, folder, filename),
                    os.path.join(self.data_dir, 'emotic', folder.replace('/', os.sep), filename)
                ]
                
                image_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        image_path = path
                        break
                
                if image_path:
                    unique_images[filename] = {
                        'path': image_path,
                        'folder': folder
                    }
                else:
                    logger.warning(f"Image not found: {filename} in folder {folder}")
        
        return unique_images
    
    def generate_captions_for_split(self, split, resume=False):
        """Generate captions for all images in a specific split"""
        logger.info(f"Processing {split} split...")
        
        # Get unique images
        unique_images = self.get_unique_images_from_csv(split)
        if not unique_images:
            logger.warning(f"No images found for {split} split")
            return {}
        
        logger.info(f"Found {len(unique_images)} unique images in {split} split")
        
        # Check for existing captions
        caption_file = os.path.join(self.emotic_pre_dir, f'{split}_captions.json')
        existing_captions = {}
        
        if resume and os.path.exists(caption_file):
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    existing_captions = json.load(f)
                logger.info(f"Found {len(existing_captions)} existing captions for {split}")
            except Exception as e:
                logger.warning(f"Error loading existing captions: {e}")
        
        # Generate captions
        captions = {}
        failed_count = 0
        
        for filename, info in tqdm(unique_images.items(), desc=f"Generating {split} captions with Moondream2"):
            # Skip if already exists
            if filename in existing_captions:
                captions[filename] = existing_captions[filename]
                continue
            
            # Generate caption
            caption = self.generate_caption(info['path'])
            if caption:
                captions[filename] = {
                    'caption': caption,
                    'folder': info['folder'],
                    'status': 'success'
                }
                logger.info(f"Generated: {filename} -> {caption}")
            else:
                captions[filename] = {
                    'caption': '',
                    'folder': info['folder'],
                    'status': 'failed'
                }
                failed_count += 1
            
            # Save periodically
            if len(captions) % 10 == 0:
                with open(caption_file, 'w', encoding='utf-8') as f:
                    json.dump(captions, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved checkpoint: {len(captions)} captions processed")
        
        # Save final captions
        with open(caption_file, 'w', encoding='utf-8') as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        
        # Create text descriptions array matching the CSV order
        self.create_text_descriptions_npy(split, captions)
        
        logger.info(f"Completed {split} split: {len(captions) - failed_count}/{len(unique_images)} successful")
        return captions
    
    def create_text_descriptions_npy(self, split, captions):
        """Create text_descriptions.npy file matching the CSV order"""
        if split not in self.datasets:
            logger.error(f"Split {split} not found in datasets")
            return
            
        df = self.datasets[split]
        text_descriptions = []
        
        # Create descriptions array in the same order as CSV rows
        for _, row in df.iterrows():
            filename = row['Filename']
            
            if filename in captions:
                caption_data = captions[filename]
                if isinstance(caption_data, dict):
                    caption_text = caption_data.get('caption', '')
                else:
                    caption_text = str(caption_data)
            else:
                caption_text = ''  # Empty string for missing captions
            
            text_descriptions.append(caption_text)
        
        # Convert to numpy array
        text_descriptions_array = np.array(text_descriptions, dtype=object)
        
        # Save as npy file
        npy_path = os.path.join(self.emotic_pre_dir, f'{split}_text_descriptions.npy')
        np.save(npy_path, text_descriptions_array)
        
        logger.info(f"Saved {split}_text_descriptions.npy with {len(text_descriptions_array)} descriptions")
        
        # Save statistics
        stats = {
            'total_samples': len(text_descriptions_array),
            'samples_with_captions': len([desc for desc in text_descriptions_array if desc and desc.strip()]),
            'samples_without_captions': len([desc for desc in text_descriptions_array if not desc or not desc.strip()]),
            'average_caption_length': np.mean([len(desc.split()) for desc in text_descriptions_array if desc and desc.strip()]) if any(text_descriptions_array) else 0,
            'model_used': 'moondream2'
        }
        
        stats_path = os.path.join(self.emotic_pre_dir, f'{split}_text_descriptions_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Moondream2 text descriptions stats for {split}: {stats}")
        
        return text_descriptions_array
    
    def sample_captions(self, split='train', num_samples=5):
        """Generate captions for a small sample"""
        if split not in self.datasets:
            logger.error(f"Split {split} not found")
            return
        
        unique_images = self.get_unique_images_from_csv(split)
        sample_images = dict(list(unique_images.items())[:num_samples])
        
        logger.info(f"Generating sample captions for {len(sample_images)} images from {split} split using Moondream2...")
        
        sample_results = {}
        for filename, info in tqdm(sample_images.items(), desc="Sample generation with Moondream2"):
            caption = self.generate_caption(info['path'])
            sample_results[filename] = {
                'caption': caption,
                'folder': info['folder']
            }
            logger.info(f"{filename}: {caption}")
        
        # Save sample results
        sample_file = os.path.join(self.emotic_pre_dir, f'sample_captions_{split}_moondream2.json')
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_results, f, indent=2, ensure_ascii=False)
        
        return sample_results


def main():
    parser = argparse.ArgumentParser(description="Generate captions using Moondream2 model (requires Python 3.10+)")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--emotic_pre_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--model_revision', type=str, default='2025-06-21')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--sample_only', action='store_true')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--length', type=str, default='short', choices=['short', 'normal', 'long'], 
                       help='Caption length: short, normal, or long')
    
    args = parser.parse_args()
    
    # Initialize caption generator
    generator = Moondream2CaptionGenerator(
        data_dir=args.data_dir,
        emotic_pre_dir=args.emotic_pre_dir,
        model_revision=args.model_revision,
        device=args.device,
        caption_length=args.length
    )
    
    # Generate captions
    if args.sample_only:
        for split in args.splits:
            if split in generator.datasets:
                generator.sample_captions(split=split, num_samples=args.num_samples)
    else:
        for split in args.splits:
            if split in generator.datasets:
                generator.generate_captions_for_split(split, resume=args.resume)
    
    logger.info("Moondream2 caption generation completed!")


if __name__ == "__main__":
    main()