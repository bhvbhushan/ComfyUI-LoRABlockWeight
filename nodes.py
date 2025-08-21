import torch
import json
import os
import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import folder_paths
import comfy.model_management
import comfy.utils
import comfy.sd
import comfy.lora

class NunchakuHierarchicalLoRALoader:
    """
    Advanced LoRA loader with per-block weight control. Apply different LoRA strengths
    to specific transformer blocks for fine-grained control over model behavior.
    Compatible with Flux, Nunchaku, and standard Stable Diffusion models.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "block_weights": ("STRING", {
                    "multiline": True, 
                    "default": "1.0",
                    "placeholder": "Enter weights: single value, list, or block-specific"
                }),
                "weight_mode": ([
                    "uniform", 
                    "block_specific", 
                    "linear_interpolation", 
                    "exponential", 
                    "gaussian",
                    "custom_curve"
                ],),
                "interpolation_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "interpolation_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "interpolation_curve": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "block_range": ("STRING", {"default": "all", "placeholder": "all, 0-6, 7-12, 13-18, or custom"}),
                "normalize_weights": ("BOOLEAN", {"default": False}),
                "preserve_mean": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": False}),
                "preset": (["none"] + cls.get_available_presets(),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "load_lora"
    CATEGORY = "loaders/advanced"
    DESCRIPTION = "Load LoRA with per-block weight control for fine-grained influence over different model layers"

    @classmethod
    def get_available_presets(cls) -> List[str]:
        """Dynamically load available presets"""
        presets = [
            "linear_decay",
            "linear_growth", 
            "bell_curve",
            "u_shape",
            "emphasis_early",
            "emphasis_late",
            "emphasis_middle"
        ]
        
        # Load custom presets from directory
        preset_dir = Path(__file__).parent / "presets"
        if preset_dir.exists():
            for file in preset_dir.glob("*.json"):
                preset_name = file.stem
                if preset_name not in presets:
                    presets.append(preset_name)
        
        return sorted(presets)

    def __init__(self):
        self.preset_functions = {
            "linear_decay": lambda n: np.linspace(1.5, 0.5, n),
            "linear_growth": lambda n: np.linspace(0.5, 1.5, n),
            "bell_curve": lambda n: self.generate_bell_curve(n, 1.5, 0.5),
            "u_shape": lambda n: self.generate_u_curve(n, 0.5, 1.5),
            "emphasis_early": lambda n: self.generate_emphasis_curve(n, "early"),
            "emphasis_late": lambda n: self.generate_emphasis_curve(n, "late"),
            "emphasis_middle": lambda n: self.generate_emphasis_curve(n, "middle"),
        }

    def generate_bell_curve(self, n: int, peak: float, trough: float) -> np.ndarray:
        """Generate bell-shaped weight distribution"""
        x = np.linspace(-3, 3, n)
        weights = np.exp(-x**2 / 2)
        return trough + (peak - trough) * weights / weights.max()

    def generate_u_curve(self, n: int, trough: float, peak: float) -> np.ndarray:
        """Generate U-shaped weight distribution"""
        x = np.linspace(-1, 1, n)
        weights = x**2
        return trough + (peak - trough) * weights

    def generate_emphasis_curve(self, n: int, position: str) -> np.ndarray:
        """Generate emphasis curve for early/middle/late blocks"""
        if position == "early":
            return np.concatenate([np.ones(n//3) * 1.5, np.ones(n - n//3) * 0.75])
        elif position == "late":
            return np.concatenate([np.ones(n - n//3) * 0.75, np.ones(n//3) * 1.5])
        else:  # middle
            third = n // 3
            return np.concatenate([
                np.ones(third) * 0.75,
                np.ones(third) * 1.5,
                np.ones(n - 2*third) * 0.75
            ])

    def parse_block_range(self, range_str: str, total_blocks: int) -> List[int]:
        """Parse block range specification into list of block indices"""
        range_str = range_str.strip().lower()
        
        if range_str == "all":
            return list(range(total_blocks))
        
        # Parse range patterns like "0-6", "7-12,15-18"
        blocks = []
        parts = range_str.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                try:
                    start_idx = int(start.strip())
                    end_idx = int(end.strip())
                    blocks.extend(range(start_idx, min(end_idx + 1, total_blocks)))
                except ValueError:
                    continue
            else:
                try:
                    idx = int(part)
                    if 0 <= idx < total_blocks:
                        blocks.append(idx)
                except ValueError:
                    continue
        
        return sorted(set(blocks)) if blocks else list(range(total_blocks))

    def parse_weights(self, weight_string: str, block_count: int) -> List[float]:
        """Parse weight input with support for various formats"""
        weight_string = weight_string.strip()
        
        # Handle empty or default
        if not weight_string or weight_string == "1.0":
            return [1.0] * block_count
        
        try:
            # Try to parse as JSON first
            if weight_string.startswith('[') or weight_string.startswith('{'):
                weights = json.loads(weight_string)
                if isinstance(weights, dict):
                    # Handle dict with block indices as keys
                    result = [1.0] * block_count
                    for key, value in weights.items():
                        if isinstance(key, str) and key.isdigit():
                            idx = int(key)
                            if 0 <= idx < block_count:
                                result[idx] = float(value)
                    return result
                else:
                    weights = [float(w) for w in weights]
            # Handle comma or space separated
            elif ',' in weight_string:
                weights = [float(w.strip()) for w in weight_string.split(',') if w.strip()]
            elif ' ' in weight_string:
                weights = [float(w) for w in weight_string.split() if w]
            # Handle single value
            else:
                single_weight = float(weight_string)
                return [single_weight] * block_count
            
            # Pad or truncate to match block count
            if len(weights) < block_count:
                # Repeat last value or use 1.0
                last_val = weights[-1] if weights else 1.0
                weights.extend([last_val] * (block_count - len(weights)))
            elif len(weights) > block_count:
                weights = weights[:block_count]
            
            return weights
            
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Warning: Could not parse weights '{weight_string}': {e}")
            return [1.0] * block_count

    def generate_interpolated_weights(
        self, 
        block_count: int, 
        mode: str, 
        start: float, 
        end: float, 
        curve: float
    ) -> List[float]:
        """Generate interpolated weights based on mode"""
        if mode == "linear_interpolation":
            return np.linspace(start, end, block_count).tolist()
        
        elif mode == "exponential":
            x = np.linspace(0, 1, block_count)
            if curve == 1.0:
                weights = start + (end - start) * x
            else:
                weights = start + (end - start) * (x ** curve)
            return weights.tolist()
        
        elif mode == "gaussian":
            x = np.linspace(-3, 3, block_count)
            gaussian = np.exp(-x**2 / (2 * curve))
            weights = start + (end - start) * gaussian / gaussian.max()
            return weights.tolist()
        
        else:
            return [1.0] * block_count

    def detect_model_architecture(self, model) -> Dict[str, Any]:
        """Detect model architecture and capabilities"""
        architecture = {
            "type": "unknown",
            "has_native_lora": False,
            "quantization_type": None,
            "double_blocks": 0,
            "single_blocks": 0,
            "total_blocks": 0,
            "block_names": [],
            "supports_hierarchical": False,
            "model_size_gb": 0
        }
        
        try:
            # Check for Flux/Nunchaku architecture
            if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                diffusion_model = model.model.diffusion_model
                
                # Detect Flux double/single block structure
                if hasattr(diffusion_model, 'double_blocks'):
                    architecture["type"] = "flux"
                    architecture["supports_hierarchical"] = True
                    
                    # Count double blocks with validation
                    if hasattr(diffusion_model.double_blocks, '__len__'):
                        detected_count = len(diffusion_model.double_blocks)
                        architecture["double_blocks"] = detected_count
                        # Validate against known Flux variants
                        if detected_count not in [19, 22, 24]:  # Known Flux variants
                            print(f"Warning: Unusual double_blocks count {detected_count}, may need adjustment")
                    else:
                        architecture["double_blocks"] = 19  # Flux default
                    
                    # Count single blocks with validation
                    if hasattr(diffusion_model, 'single_blocks'):
                        if hasattr(diffusion_model.single_blocks, '__len__'):
                            detected_count = len(diffusion_model.single_blocks)
                            architecture["single_blocks"] = detected_count
                            # Validate against known Flux variants
                            if detected_count not in [38, 44, 48]:  # Known Flux variants
                                print(f"Warning: Unusual single_blocks count {detected_count}, may need adjustment")
                        else:
                            architecture["single_blocks"] = 38  # Flux default
                
                # Check for native LoRA support (Nunchaku/quantized models)
                # Proper detection: check for actual Nunchaku signatures
                if hasattr(model, 'model'):
                    model_obj = model.model
                    # Check for Nunchaku-specific attributes
                    if hasattr(model_obj, 'quantization_config') or \
                       hasattr(model_obj, 'quantized_layers') or \
                       hasattr(model_obj, 'nunchaku_version'):
                        architecture["has_native_lora"] = True
                        architecture["quantization_type"] = "nunchaku"
                    # Check for generic load_lora_weights support
                    elif hasattr(model, 'load_lora_weights') or \
                         (hasattr(model_obj, 'load_lora_weights')):
                        architecture["has_native_lora"] = True
                        architecture["quantization_type"] = "unknown"
                
                # Calculate total blocks
                architecture["total_blocks"] = architecture["double_blocks"] + architecture["single_blocks"]
                
                # Generate block names for reference
                for i in range(architecture["double_blocks"]):
                    architecture["block_names"].append(f"double_block.{i}")
                for i in range(architecture["single_blocks"]):
                    architecture["block_names"].append(f"single_block.{i}")
            
            # Fallback for other architectures
            if architecture["type"] == "unknown":
                # Try to detect SD1.5/SDXL style models
                if hasattr(model, 'model'):
                    model_obj = model.model
                    if hasattr(model_obj, 'input_blocks') or hasattr(model_obj, 'model_channels'):
                        architecture["type"] = "stable_diffusion"
                        # Estimate block count for SD models
                        architecture["total_blocks"] = 25  # Typical for SD models
                        architecture["supports_hierarchical"] = True
                    
        except Exception as e:
            print(f"Architecture detection warning: {e}")
        
        return architecture

    def apply_hierarchical_weights(
        self, 
        lora_weights: Dict, 
        block_weights: List[float], 
        architecture: Dict,
        block_indices: List[int]
    ) -> Dict:
        """Apply hierarchical weights to specific LoRA tensor blocks"""
        weighted_lora = {}
        applied_blocks = set()
        
        for key, tensor in lora_weights.items():
            weight_applied = False
            key_lower = key.lower()
            
            # Handle Flux architecture
            if architecture["type"] == "flux":
                # Check for double blocks
                if "double_block" in key_lower:
                    match = re.search(r'double_blocks?\.(\d+)', key_lower)
                    if match:
                        block_idx = int(match.group(1))
                        if block_idx in block_indices and block_idx < len(block_weights):
                            weight = block_weights[block_idx]
                            weighted_lora[key] = tensor * weight
                            applied_blocks.add(f"double_block.{block_idx}")
                            weight_applied = True
                
                # Check for single blocks
                elif "single_block" in key_lower:
                    match = re.search(r'single_blocks?\.(\d+)', key_lower)
                    if match:
                        raw_idx = int(match.group(1))
                        # Adjust index for single blocks (comes after double blocks)
                        block_idx = architecture["double_blocks"] + raw_idx
                        if block_idx in block_indices and block_idx < len(block_weights):
                            weight = block_weights[block_idx]
                            weighted_lora[key] = tensor * weight
                            applied_blocks.add(f"single_block.{raw_idx}")
                            weight_applied = True
            
            # Handle other architectures with numbered blocks
            else:
                block_patterns = [
                    r'block\.(\d+)',
                    r'input_blocks\.(\d+)',
                    r'output_blocks\.(\d+)',
                    r'middle_block\.(\d+)',
                    r'layer\.(\d+)',
                    r'blocks\.(\d+)',
                ]
                
                for pattern in block_patterns:
                    match = re.search(pattern, key_lower)
                    if match:
                        block_idx = int(match.group(1))
                        if block_idx in block_indices and block_idx < len(block_weights):
                            weight = block_weights[block_idx]
                            weighted_lora[key] = tensor * weight
                            applied_blocks.add(f"block.{block_idx}")
                            weight_applied = True
                            break
            
            # If no specific block pattern matched, apply default weight
            if not weight_applied:
                # Use average weight for non-block tensors
                avg_weight = np.mean([block_weights[i] for i in block_indices]) if block_indices else 1.0
                weighted_lora[key] = tensor * avg_weight
        
        return weighted_lora

    def normalize_weights(self, weights: List[float], preserve_mean: bool = True) -> List[float]:
        """Normalize weights to preserve overall strength"""
        if not weights:
            return weights
        
        weights = np.array(weights)
        
        if preserve_mean:
            # Preserve the mean value
            current_mean = weights.mean()
            if current_mean > 0:
                target_mean = 1.0
                weights = weights * (target_mean / current_mean)
        else:
            # Normalize to [0, 1] range
            min_w, max_w = weights.min(), weights.max()
            if max_w > min_w:
                weights = (weights - min_w) / (max_w - min_w)
        
        return weights.tolist()

    def load_custom_preset(self, preset_name: str) -> Optional[Dict]:
        """Load custom preset from file with security validation"""
        # Validate preset name to prevent path traversal
        if '..' in preset_name or '/' in preset_name or '\\' in preset_name:
            print(f"Invalid preset name: {preset_name}")
            return None
        
        preset_dir = Path(__file__).parent / "presets"
        preset_file = preset_dir / f"{preset_name}.json"
        
        if preset_file.exists():
            try:
                with open(preset_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading preset {preset_name}: {e}")
        
        return None

    def generate_info_text(
        self, 
        architecture: Dict, 
        weights: List[float], 
        block_indices: List[int],
        mode: str,
        preset: str
    ) -> str:
        """Generate informative text about the weight application"""
        info = []
        
        # Architecture info
        info.append(f"Architecture: {architecture['type'].upper()}")
        if architecture['supports_hierarchical']:
            info.append(f"Hierarchical Support: Yes")
        if architecture['has_native_lora']:
            info.append(f"Native LoRA: Yes")
        
        # Block structure
        if architecture['type'] == 'flux':
            info.append(f"Double Blocks: {architecture['double_blocks']}")
            info.append(f"Single Blocks: {architecture['single_blocks']}")
        info.append(f"Total Blocks: {architecture['total_blocks']}")
        
        # Weight info
        info.append(f"\nWeight Mode: {mode}")
        if preset != "none":
            info.append(f"Preset: {preset}")
        
        # Block range
        if len(block_indices) == architecture['total_blocks']:
            info.append(f"Block Range: All blocks")
        else:
            info.append(f"Block Range: {len(block_indices)} blocks selected")
        
        # Weight statistics
        weights_array = np.array(weights)
        info.append(f"\nWeight Statistics:")
        info.append(f"  Min: {weights_array.min():.3f}")
        info.append(f"  Max: {weights_array.max():.3f}")
        info.append(f"  Mean: {weights_array.mean():.3f}")
        info.append(f"  Std: {weights_array.std():.3f}")
        
        # Weight distribution preview (compact)
        if len(weights) > 10:
            preview = weights[:5] + ['...'] + weights[-5:]
            info.append(f"\nWeights: {preview}")
        else:
            info.append(f"\nWeights: {weights}")
        
        return "\n".join(info)

    def load_lora(
        self, 
        model, 
        clip, 
        lora_name, 
        strength_model, 
        strength_clip,
        block_weights="1.0",
        weight_mode="uniform",
        interpolation_start=1.0,
        interpolation_end=1.0,
        interpolation_curve=1.0,
        block_range="all",
        normalize_weights=False,
        preserve_mean=True,
        verbose=False,
        preset="none"
    ):
        """Load LoRA with hierarchical block weight control"""
        
        # Detect model architecture
        architecture = self.detect_model_architecture(model)
        
        if not architecture['supports_hierarchical'] and weight_mode != "uniform":
            print(f"Warning: Model type '{architecture['type']}' may not fully support hierarchical weights")
        
        # Parse block range
        block_indices = self.parse_block_range(block_range, architecture['total_blocks'])
        
        # Generate or parse weights
        if preset != "none":
            # Use preset
            if preset in self.preset_functions:
                weights = self.preset_functions[preset](architecture['total_blocks']).tolist()
            else:
                # Try loading custom preset
                custom_preset = self.load_custom_preset(preset)
                if custom_preset:
                    if 'weights' in custom_preset:
                        preset_weights = custom_preset['weights']
                        if isinstance(preset_weights, dict):
                            # Handle Flux-specific presets
                            weights = []
                            if 'double_blocks' in preset_weights:
                                weights.extend(preset_weights['double_blocks'])
                            if 'single_blocks' in preset_weights:
                                weights.extend(preset_weights['single_blocks'])
                        else:
                            weights = preset_weights
                    else:
                        weights = [1.0] * architecture['total_blocks']
                else:
                    weights = [1.0] * architecture['total_blocks']
        elif weight_mode in ["linear_interpolation", "exponential", "gaussian"]:
            # Generate interpolated weights
            weights = self.generate_interpolated_weights(
                architecture['total_blocks'],
                weight_mode,
                interpolation_start,
                interpolation_end,
                interpolation_curve
            )
        elif weight_mode == "custom_curve" or weight_mode == "block_specific":
            # Parse custom weights
            weights = self.parse_weights(block_weights, architecture['total_blocks'])
        else:
            # Uniform weights
            weights = self.parse_weights(block_weights, architecture['total_blocks'])
        
        # Apply normalization if requested
        if normalize_weights:
            weights = self.normalize_weights(weights, preserve_mean)
        
        # Load LoRA file with size check
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise ValueError(f"LoRA file '{lora_name}' not found")
        
        # Check file size for memory planning
        import os
        file_size_gb = os.path.getsize(lora_path) / (1024**3)
        if file_size_gb > 2.0:
            print(f"Large LoRA file ({file_size_gb:.2f}GB), may require extra memory")
        
        try:
            # Load LoRA weights
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            
            # Apply hierarchical weights if supported
            if architecture['supports_hierarchical'] and weight_mode != "uniform":
                weighted_lora = self.apply_hierarchical_weights(
                    lora_data, 
                    weights, 
                    architecture,
                    block_indices
                )
            else:
                # Apply uniform weight
                uniform_weight = np.mean(weights)
                weighted_lora = {}
                # Apply weights with memory efficiency
                for k, v in lora_data.items():
                    weighted_lora[k] = v * uniform_weight
                    # Clear original tensor if no longer needed
                    if k in weighted_lora:
                        del lora_data[k]
            
            # Load into model with proper architecture handling
            if architecture['has_native_lora']:
                # Try native LoRA loading first
                try:
                    if hasattr(model, 'load_lora_weights'):
                        # Direct model method
                        model_lora = model.load_lora_weights(weighted_lora, strength_model)
                        clip_lora = clip  # CLIP handled separately in some cases
                    else:
                        # Use ComfyUI's LoRA loader with weighted tensors
                        model_lora, clip_lora = comfy.lora.load_lora(
                            weighted_lora, model, clip, strength_model, strength_clip
                        )
                except Exception as native_error:
                    print(f"Native loading failed: {native_error}, using standard method")
                    model_lora, clip_lora = comfy.lora.load_lora(
                        weighted_lora, model, clip, strength_model, strength_clip
                    )
            else:
                # Standard ComfyUI LoRA loading with weighted tensors
                model_lora, clip_lora = comfy.lora.load_lora(
                    weighted_lora, model, clip, strength_model, strength_clip
                )
            
            # Memory cleanup for large models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error with hierarchical loading: {e}")
            print("Falling back to weighted LoRA loading...")
            # Fallback but still use weighted tensors!
            try:
                # Load raw LoRA data
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                # Apply uniform weight as fallback
                avg_weight = np.mean(weights) if weights else 1.0
                weighted_lora = {k: v * avg_weight for k, v in lora_data.items()}
                # Load weighted tensors
                model_lora, clip_lora = comfy.lora.load_lora(
                    weighted_lora, model, clip, strength_model, strength_clip
                )
            except:
                # Last resort: standard loading
                print("Final fallback to standard LoRA loading")
                model_lora, clip_lora = comfy.lora.load_lora_for_models(
                    model, clip, lora_path, strength_model, strength_clip
                )
        
        # Generate info text
        if verbose:
            info = self.generate_info_text(architecture, weights, block_indices, weight_mode, preset)
        else:
            info = f"LoRA loaded with {weight_mode} weights"
        
        return (model_lora, clip_lora, info)


# Additional utility node for weight visualization/editing
class HierarchicalLoRAWeightEditor:
    """Helper node for creating and visualizing hierarchical weight patterns"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_blocks": ("INT", {"default": 57, "min": 1, "max": 200, "step": 1}),
                "pattern": ([
                    "uniform",
                    "linear_increase", 
                    "linear_decrease",
                    "sine_wave",
                    "cosine_wave",
                    "sawtooth",
                    "step_function",
                    "random",
                    "custom_expression"
                ],),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "offset": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "phase": ("FLOAT", {"default": 0.0, "min": -3.14159, "max": 3.14159, "step": 0.01}),
            },
            "optional": {
                "custom_expression": ("STRING", {
                    "default": "1.0 + 0.5 * sin(2 * pi * i / n)",
                    "multiline": False
                }),
                "visualize": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("weights", "visualization")
    FUNCTION = "generate_weights"
    CATEGORY = "loaders/advanced"
    DESCRIPTION = "Generate and visualize weight patterns for hierarchical LoRA loading"
    
    def generate_weights(
        self,
        total_blocks,
        pattern,
        amplitude,
        offset,
        frequency,
        phase,
        custom_expression="",
        visualize=True
    ):
        """Generate weight pattern based on parameters"""
        
        n = total_blocks
        i_values = np.arange(n)
        
        if pattern == "uniform":
            weights = np.ones(n) * offset
        elif pattern == "linear_increase":
            weights = offset + amplitude * i_values / (n - 1)
        elif pattern == "linear_decrease":
            weights = offset + amplitude * (1 - i_values / (n - 1))
        elif pattern == "sine_wave":
            weights = offset + amplitude * np.sin(2 * np.pi * frequency * i_values / n + phase)
        elif pattern == "cosine_wave":
            weights = offset + amplitude * np.cos(2 * np.pi * frequency * i_values / n + phase)
        elif pattern == "sawtooth":
            weights = offset + amplitude * ((frequency * i_values / n + phase / (2 * np.pi)) % 1)
        elif pattern == "step_function":
            step_size = max(1, int(n / frequency))
            weights = offset + amplitude * ((i_values // step_size) % 2)
        elif pattern == "random":
            np.random.seed(int(phase * 1000))
            weights = offset + amplitude * np.random.random(n)
        elif pattern == "custom_expression" and custom_expression:
            try:
                # Safe evaluation using numexpr or simpler math parser
                import ast
                import operator as op
                
                # Define safe operations
                safe_ops = {
                    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                    ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
                    ast.Mod: op.mod
                }
                
                safe_funcs = {
                    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                    'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                    'abs': np.abs, 'min': np.minimum, 'max': np.maximum
                }
                
                def safe_eval(node, variables):
                    if isinstance(node, ast.Num):
                        return node.n
                    elif isinstance(node, ast.Name):
                        if node.id in variables:
                            return variables[node.id]
                        elif node.id in ['pi', 'e']:
                            return np.pi if node.id == 'pi' else np.e
                        raise ValueError(f"Unknown variable: {node.id}")
                    elif isinstance(node, ast.BinOp):
                        return safe_ops[type(node.op)](
                            safe_eval(node.left, variables),
                            safe_eval(node.right, variables)
                        )
                    elif isinstance(node, ast.UnaryOp):
                        return safe_ops[type(node.op)](safe_eval(node.operand, variables))
                    elif isinstance(node, ast.Call):
                        if node.func.id in safe_funcs:
                            args = [safe_eval(arg, variables) for arg in node.args]
                            return safe_funcs[node.func.id](*args)
                        raise ValueError(f"Unknown function: {node.func.id}")
                    else:
                        raise ValueError(f"Unsupported operation: {type(node)}")
                
                # Parse and evaluate expression safely
                tree = ast.parse(custom_expression, mode='eval')
                variables = {'i': i_values, 'n': n}
                weights = safe_eval(tree.body, variables)
                
                if not isinstance(weights, np.ndarray):
                    weights = np.array([weights] * n)
            except Exception as e:
                print(f"Error evaluating expression: {e}")
                weights = np.ones(n) * offset
        else:
            weights = np.ones(n) * offset
        
        # Clamp weights to reasonable range
        weights = np.clip(weights, 0.0, 2.0)
        
        # Format weights as string
        weight_str = ", ".join([f"{w:.3f}" for w in weights])
        
        # Generate visualization
        if visualize:
            viz = self.create_ascii_visualization(weights)
        else:
            viz = "Visualization disabled"
        
        return (weight_str, viz)
    
    def create_ascii_visualization(self, weights: np.ndarray) -> str:
        """Create ASCII art visualization of weight distribution"""
        
        height = 10
        width = min(len(weights), 60)
        
        # Resample if needed
        if len(weights) > width:
            indices = np.linspace(0, len(weights) - 1, width).astype(int)
            display_weights = weights[indices]
        else:
            display_weights = weights
        
        # Normalize to 0-height range
        min_w, max_w = display_weights.min(), display_weights.max()
        if max_w > min_w:
            normalized = (display_weights - min_w) / (max_w - min_w)
        else:
            normalized = np.ones_like(display_weights) * 0.5
        
        # Create grid
        grid = []
        for h in range(height, 0, -1):
            row = []
            threshold = h / height
            for w in normalized:
                if w >= threshold:
                    row.append('█')
                elif w >= threshold - 0.1:
                    row.append('▄')
                else:
                    row.append(' ')
            grid.append(''.join(row))
        
        # Add axis
        grid.append('─' * width)
        
        # Add labels
        viz = [
            f"Weight Distribution (blocks: {len(weights)})",
            f"Range: [{weights.min():.3f}, {weights.max():.3f}]",
            f"Mean: {weights.mean():.3f}, Std: {weights.std():.3f}",
            "",
        ]
        viz.extend(grid)
        
        return "\n".join(viz)


NODE_CLASS_MAPPINGS = {
    "NunchakuHierarchicalLoRALoader": NunchakuHierarchicalLoRALoader,
    "HierarchicalLoRAWeightEditor": HierarchicalLoRAWeightEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuHierarchicalLoRALoader": "LoRA Block Weight Loader",
    "HierarchicalLoRAWeightEditor": "LoRA Block Weight Editor",
}