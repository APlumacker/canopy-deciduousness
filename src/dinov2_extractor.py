#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv2 feature extraction for GeoTIFF orthomosaics: sliding-window tiling
(tile size multiple of 14, overlap), batch inference, overlap averaging,
optional vector mask, output one raster per orthomosaic with one band per
feature dimension (384 bands with the default dinov2_vits14 model), i.e. one
value per 14 x 14-pixel patch.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import gc

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
import fiona
from shapely.geometry import shape
from pyproj import Transformer
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


@dataclass
class TileInfo:
    """Information about a tile to process."""
    tx: int
    ty: int
    col_off: int
    row_off: int
    width: int
    height: int
    out_col: int
    out_row: int
    out_width: int
    out_height: int


# Model -> feature dimension mapping
MODEL_DIMENSIONS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}


class DINOv2FeatureExtractor:
    """GPU-optimised DINOv2 feature extractor."""
    
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: Optional[str] = None,
        use_compile: bool = True
    ):
        """
        Initialise the DINOv2 extractor.
        
        Args:
            model_name: Model name ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
            device: PyTorch device ('cuda', 'cpu', or None for auto-detection)
            use_compile: Use torch.compile for speed-up (PyTorch 2.0+)
        """
        if model_name not in MODEL_DIMENSIONS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Choose from: {list(MODEL_DIMENSIONS.keys())}"
            )
        
        self.model_name = model_name
        
        # Automatic device detection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        print(f"Selected model: {model_name} ({MODEL_DIMENSIONS[model_name]} dimensions)")
        
        # CUDA optimisations if available
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Available GPU memory: {mem_gb:.2f} GB")
            
            # Optimisations for recent GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Batch size recommendations
            if model_name == 'dinov2_vits14':
                if mem_gb < 4:
                    print("Recommended: --batch-size 4")
                elif mem_gb < 8:
                    print("Recommended: --batch-size 8-12")
                else:
                    print("Recommended: --batch-size 16-24")
            elif model_name == 'dinov2_vitb14':
                if mem_gb < 8:
                    print("Recommended: --batch-size 2-4")
                else:
                    print("Recommended: --batch-size 4-8")
        
        # Load the DINOv2 model
        print(f"Loading model {model_name}...")
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Model compilation (PyTorch 2.0+)
            if use_compile and hasattr(torch, 'compile') and self.device.type == "cuda":
                print("Compiling model (may take ~1 min)...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled")
        except Exception as e:
            raise RuntimeError(f"Error while loading model: {e}")
        
        # Model parameters
        self.patch_size = 14
        self.feature_dim = MODEL_DIMENSIONS[model_name]
        
        # Image normalisation
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def preprocess_tiles_batch(self, tiles: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a batch of tiles with automatic padding."""
        max_h = max(tile.shape[0] for tile in tiles)
        max_w = max(tile.shape[1] for tile in tiles)
        
        batch_tensors = []
        for tile in tiles:
            tile_float = tile.astype(np.float32) / 255.0
            tile_tensor = torch.from_numpy(tile_float).permute(2, 0, 1)
            tile_tensor = self.transform(tile_tensor)
            
            if tile.shape[0] < max_h or tile.shape[1] < max_w:
                pad_h = max_h - tile.shape[0]
                pad_w = max_w - tile.shape[1]
                tile_tensor = F.pad(tile_tensor, (0, pad_w, 0, pad_h), value=0)
            
            batch_tensors.append(tile_tensor)
        
        return torch.stack(batch_tensors)
    
    @torch.no_grad()
    def extract_features_batch(self, tiles: List[np.ndarray]) -> List[np.ndarray]:
        """Extract DINOv2 features from a batch of tiles."""
        if not tiles:
            return []
        
        batch_tensor = self.preprocess_tiles_batch(tiles).to(self.device, non_blocking=True)
        features = self.model.forward_features(batch_tensor)
        patch_tokens = features['x_norm_patchtokens']
        
        results = []
        padded_h = batch_tensor.shape[2]
        padded_w = batch_tensor.shape[3]
        padded_h_patches = padded_h // self.patch_size
        padded_w_patches = padded_w // self.patch_size
        
        for i, tile in enumerate(tiles):
            h_patches = tile.shape[0] // self.patch_size
            w_patches = tile.shape[1] // self.patch_size
            
            tile_tokens = patch_tokens[i]
            tile_tokens_grid = tile_tokens.reshape(padded_h_patches, padded_w_patches, self.feature_dim)
            features_grid = tile_tokens_grid[:h_patches, :w_patches, :]
            features_np = features_grid.cpu().numpy()
            results.append(features_np)
        
        return results


def prepare_tiles_with_overlap(
    src: rasterio.DatasetReader,
    tile_size: int,
    overlap: int,
    patch_size: int
) -> List[TileInfo]:
    """Build the list of overlapping tiles for the sliding window."""
    step = tile_size - overlap
    tiles = []
    
    for ty in range(0, src.height, step):
        for tx in range(0, src.width, step):
            col_off = tx
            row_off = ty
            
            width = min(tile_size, src.width - col_off)
            height = min(tile_size, src.height - row_off)
            
            width = (width // patch_size) * patch_size
            height = (height // patch_size) * patch_size
            
            if width < patch_size or height < patch_size:
                continue
            
            out_col = col_off // patch_size
            out_row = row_off // patch_size
            out_width = width // patch_size
            out_height = height // patch_size
            
            tiles.append(TileInfo(
                tx=tx // step, ty=ty // step,
                col_off=col_off, row_off=row_off,
                width=width, height=height,
                out_col=out_col, out_row=out_row,
                out_width=out_width, out_height=out_height
            ))
    
    return tiles


def read_geotiff_metadata(input_path: str) -> Dict:
    """Read GeoTIFF metadata."""
    with rasterio.open(input_path) as src:
        return {
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': src.dtypes[0],
            'nodata': src.nodata
        }


def load_vector_mask(
    mask_path: str,
    raster_transform,
    raster_shape: Tuple[int, int],
    raster_crs
) -> np.ndarray:
    """Load a vector mask and rasterise it."""
    print(f"\nLoading vector mask: {mask_path}")
    
    geometries = []
    with fiona.open(mask_path) as src:
        mask_crs = src.crs
        print(f"  - Mask CRS: {mask_crs}")
        print(f"  - Number of features: {len(src)}")

        need_reproject = (
            mask_crs is not None
            and raster_crs is not None
            and mask_crs != raster_crs
        )
        if need_reproject:
            print(f"  Different CRS (raster: {raster_crs}) - reprojecting automatically")
            transformer = Transformer.from_crs(
                mask_crs, raster_crs, always_xy=True
            )
        else:
            transformer = None

        for feature in src:
            geom = shape(feature['geometry'])
            if transformer is not None:
                from shapely.ops import transform as shapely_transform
                geom = shapely_transform(transformer.transform, geom)
            geometries.append(geom)
    
    print("  - Rasterising mask...")
    mask = geometry_mask(
        geometries,
        out_shape=raster_shape,
        transform=raster_transform,
        invert=True
    )
    
    n_masked = np.sum(mask)
    total = mask.size
    pct = (n_masked / total) * 100
    
    print(f"  Mask loaded: {n_masked:,} pixels to process ({pct:.1f}%)")
    
    return mask


def reduce_mask_to_patches_vectorized(raster_mask: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Reduce a raster mask to patch resolution (vectorised).
    Uses numpy operations instead of Python loops.
    
    Args:
        raster_mask: Full-resolution binary mask (H, W)
        patch_size: Patch size (14 for DINOv2)
        
    Returns:
        Reduced mask (out_height, out_width), one value per patch
    """
    h, w = raster_mask.shape
    out_height = h // patch_size
    out_width = w // patch_size
    
    print(f"  - Reducing mask: {h}x{w} -> {out_height}x{out_width} patches (vectorised)...")
    
    # Crop to exact multiples of patch_size
    cropped_h = out_height * patch_size
    cropped_w = out_width * patch_size
    cropped_mask = raster_mask[:cropped_h, :cropped_w]
    
    # Reshape to (out_height, patch_size, out_width, patch_size), then average
    # (much faster than Python loops)
    reshaped = cropped_mask.reshape(out_height, patch_size, out_width, patch_size)
    out_mask = reshaped.mean(axis=(1, 3)) > 0.5
    
    return out_mask


def process_geotiff_with_dinov2(
    input_path: str,
    output_path: str,
    model_name: str = "dinov2_vits14",
    tile_size: int = 518,
    overlap: int = 70,
    batch_size: int = 8,
    device: Optional[str] = None,
    use_compile: bool = True,
    mask_path: Optional[str] = None,
    mask_buffer: int = 0,
    block_height: Optional[int] = None
) -> None:
    """
    Process a full GeoTIFF with DINOv2 using a sliding window.
    Everything is kept in memory and written in a single pass for optimal compression.
    
    Args:
        input_path: Path to the input GeoTIFF
        output_path: Path to the output GeoTIFF (features)
        model_name: DINOv2 model name
        tile_size: Tile size in pixels (must be a multiple of 14)
        overlap: Overlap between tiles (pixels)
        batch_size: GPU batch size
        device: PyTorch device
        use_compile: Use torch.compile
        mask_path: Path to the vector mask (optional)
        mask_buffer: Buffer around the mask (pixels)
        block_height: Ignored (kept for pipeline compatibility); processing is always in memory
    """
    if tile_size % 14 != 0:
        raise ValueError(f"tile_size must be a multiple of 14, got: {tile_size}")
    
    if overlap > 0 and overlap % 14 != 0:
        overlap_rounded = ((overlap + 7) // 14) * 14
        print(f"Overlap rounded from {overlap} to {overlap_rounded} (multiple of 14)")
        overlap = overlap_rounded
    
    # Free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialise the extractor
    extractor = DINOv2FeatureExtractor(
        model_name=model_name,
        device=device,
        use_compile=use_compile
    )
    
    # Read metadata
    print(f"\nReading image: {input_path}")
    src_metadata = read_geotiff_metadata(input_path)
    print(f"Dimensions: {src_metadata['width']} x {src_metadata['height']}")
    print(f"CRS: {src_metadata['crs']}")
    
    with rasterio.open(input_path) as src:
        if src.count not in [3, 4]:
            raise ValueError(f"Image must have 3 (RGB) or 4 (RGBA) bands, found: {src.count}")
        
        if src.count == 4:
            print(f"Alpha band detected - it will be ignored")
        
        # Output dimensions
        patch_size = extractor.patch_size
        out_height = src.height // patch_size
        out_width = src.width // patch_size
        
        print(f"Output dimensions: {out_width} x {out_height} x {extractor.feature_dim}")
        
        # Estimate required RAM
        ram_gb = (out_height * out_width * extractor.feature_dim * 4) / 1e9  # float32 = 4 bytes
        print(f"\nEstimated RAM required: ~{ram_gb:.1f} GB")
        
        if ram_gb > 32:
            print(f"WARNING: high RAM usage. If this fails, use the vits14 model or a smaller image")
        
        # Load vector mask if provided
        raster_mask = None
        if mask_path:
            raster_mask = load_vector_mask(
                mask_path,
                src.transform,
                (src.height, src.width),
                src.crs
            )
            
            # Apply buffer if requested
            if mask_buffer > 0:
                from scipy.ndimage import binary_dilation
                struct = np.ones((mask_buffer * 2 + 1, mask_buffer * 2 + 1))
                raster_mask = binary_dilation(raster_mask, structure=struct)
                print(f"  - Applied {mask_buffer}-pixel buffer")
        
        # Output mask at patch resolution (vectorised)
        out_mask = None
        if raster_mask is not None:
            out_mask = reduce_mask_to_patches_vectorized(raster_mask, patch_size)
            
            n_valid = np.sum(out_mask)
            n_total = out_mask.size
            print(f"\nPatches to process: {n_valid:,} / {n_total:,} ({n_valid/n_total*100:.1f}%)")
        
        # Geotransform
        new_transform = from_bounds(
            src.bounds.left, src.bounds.bottom,
            src.bounds.right, src.bounds.top,
            out_width, out_height
        )
        
        # Output profile
        output_profile = {
            'driver': 'GTiff',
            'height': out_height,
            'width': out_width,
            'count': extractor.feature_dim,
            'dtype': 'float32',
            'crs': src.crs,
            'transform': new_transform,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'BIGTIFF': 'YES'
        }
        
        # Prepare overlapping tiles
        tiles_info = prepare_tiles_with_overlap(src, tile_size, overlap, patch_size)
        
        # Filter tiles using the mask
        if out_mask is not None:
            print("\nFiltering tiles using the mask...")
            tiles_info_filtered = []
            for tile_info in tiles_info:
                tile_mask = out_mask[
                    tile_info.out_row:tile_info.out_row + tile_info.out_height,
                    tile_info.out_col:tile_info.out_col + tile_info.out_width
                ]
                if np.mean(tile_mask) > 0.1:
                    tiles_info_filtered.append(tile_info)
            
            print(f"  - Tiles before filtering: {len(tiles_info):,}")
            print(f"  - Tiles after filtering: {len(tiles_info_filtered):,}")
            print(f"  - Tiles skipped: {len(tiles_info) - len(tiles_info_filtered):,}")
            tiles_info = tiles_info_filtered
        
        total_tiles = len(tiles_info)
        
        print(f"\nSliding-window processing:")
        print(f"  - Number of tiles: {total_tiles}")
        print(f"  - Tile size: {tile_size}x{tile_size} pixels")
        print(f"  - Overlap: {overlap} pixels ({overlap//patch_size} patches)")
        print(f"  - Batch size: {batch_size}")
        
        if overlap > 0:
            print(f"  - Mode: sliding window with overlap averaging")
        else:
            print(f"  - Mode: no overlap (standard processing)")
        
        print(f"\nStrategy: fully in memory (fast, optimal compression)")
        
        # Accumulation buffers for overlap averaging
        feature_sum = np.zeros((out_height, out_width, extractor.feature_dim), dtype=np.float32)
        weight_sum = np.zeros((out_height, out_width), dtype=np.float32)
        
        # Batch processing
        with tqdm(total=total_tiles, desc="Feature extraction") as pbar:
            for i in range(0, len(tiles_info), batch_size):
                batch_tiles_info = tiles_info[i:i + batch_size]
                
                # Load the tiles of the batch
                batch_data = []
                batch_infos_valid = []
                
                for tile_info in batch_tiles_info:
                    try:
                        window = Window(
                            tile_info.col_off,
                            tile_info.row_off,
                            tile_info.width,
                            tile_info.height
                        )
                        tile_data = src.read([1, 2, 3], window=window)
                        tile_data = np.transpose(tile_data, (1, 2, 0))
                        
                        batch_data.append(tile_data)
                        batch_infos_valid.append(tile_info)
                    except Exception as e:
                        print(f"\nError reading tile: {e}")
                
                if not batch_data:
                    pbar.update(len(batch_tiles_info))
                    continue
                
                # Feature extraction
                try:
                    features_list = extractor.extract_features_batch(batch_data)
                    
                    # Accumulate into buffers (for overlap averaging)
                    for tile_info, features in zip(batch_infos_valid, features_list):
                        out_h, out_w = features.shape[0], features.shape[1]
                        
                        if out_h == 0 or out_w == 0:
                            continue
                        
                        # Bounds check
                        if (tile_info.out_col + out_w > out_width or 
                            tile_info.out_row + out_h > out_height):
                            continue
                        
                        # Blending weights
                        if overlap > 0:
                            weight = np.ones((out_h, out_w), dtype=np.float32)
                            overlap_patches = overlap // patch_size
                            
                            # Linear ramp within the overlap zone
                            if overlap_patches > 0:
                                for y in range(min(overlap_patches, out_h)):
                                    weight[y, :] *= (y + 1) / (overlap_patches + 1)
                                    weight[-(y+1), :] *= (y + 1) / (overlap_patches + 1)
                                for x in range(min(overlap_patches, out_w)):
                                    weight[:, x] *= (x + 1) / (overlap_patches + 1)
                                    weight[:, -(x+1)] *= (x + 1) / (overlap_patches + 1)
                        else:
                            weight = np.ones((out_h, out_w), dtype=np.float32)
                        
                        # Weighted accumulation
                        row_slice = slice(tile_info.out_row, tile_info.out_row + out_h)
                        col_slice = slice(tile_info.out_col, tile_info.out_col + out_w)
                        
                        feature_sum[row_slice, col_slice, :] += features * weight[:, :, np.newaxis]
                        weight_sum[row_slice, col_slice] += weight
                    
                except Exception as e:
                    print(f"\nError in batch: {e}")
                    import traceback
                    traceback.print_exc()
                
                pbar.update(len(batch_tiles_info))
        
        # Normalise by weights
        print("\nMerging overlaps and normalising...")
        
        # Avoid division by zero
        weight_sum[weight_sum == 0] = 1.0
        
        # Normalisation
        final_features = feature_sum / weight_sum[:, :, np.newaxis]
        
        # Free intermediate buffers
        del feature_sum, weight_sum
        gc.collect()
        
        # Apply output mask
        if out_mask is not None:
            print("Applying final mask...")
            final_features[~out_mask, :] = 0
        
        # Write the final file in a single pass (optimal compression)
        print("Writing file (optimal compression)...")
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            # Convert (H, W, D) -> (D, H, W) and write
            features_transposed = np.transpose(final_features, (2, 0, 1))
            dst.write(features_transposed)
        
        # Final cleanup
        del final_features, features_transposed
        gc.collect()
        
        print(f"\nFeatures saved: {output_path}")
        
        # Report file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
