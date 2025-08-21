# ComfyUI LoRA Block Weight Loader

A production-ready ComfyUI custom node that provides hierarchical per-block weight control for LoRA loading. Apply different LoRA strengths to specific transformer blocks for fine-grained control over model behavior. Fully compatible with ComfyUI 0.3.51+ and works with Flux, Nunchaku quantized models, SDXL, and SD 1.5 architectures.

## 🎯 Key Features

- **Hierarchical Per-Block Weight Control**: Apply different LoRA strengths to specific transformer blocks with proper weight scaling
- **ComfyUI 0.3.51+ API Compatibility**: Uses proper `comfy.lora.load_lora()` with key mapping and model patching
- **Universal Architecture Support**: Auto-detects and adapts to Flux (19 double + 38 single blocks), Nunchaku quantized models, SDXL, and SD 1.5
- **Advanced Weight Application**: Pre-tensor multiplication before LoRA loading for true hierarchical control
- **Multiple Weight Modes**: Uniform, linear interpolation, exponential, gaussian, bell curve, U-shape, and custom mathematical expressions
- **Intelligent Block Range Selection**: Target specific block ranges with pattern matching (e.g., "0-6" for lower blocks, "7-12" for middle, "13-18" for upper)
- **Performance Optimized**: Pre-compiled regex patterns, efficient tensor operations, proper memory cleanup
- **Robust Error Handling**: Multi-level fallback system ensures LoRA loads even with API changes
- **Weight Visualization**: Built-in weight editor with ASCII visualization for pattern preview
- **Custom Presets**: JSON-based preset system with Flux-specific double/single block support
- **Production Ready**: Fully tested with ComfyUI Manager, proper null checks, and validation

## 📦 Installation

### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "LoRA Block Weight"
3. Click Install

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/bhvbhushan/ComfyUI-LoRABlockWeight.git
# Restart ComfyUI
```

## 🚀 Quick Start

### Basic Usage
1. Add the "LoRA Block Weight Loader" node to your workflow
2. Connect your model and CLIP
3. Select a LoRA file
4. Choose a weight mode or preset
5. Adjust strengths and connect to sampling nodes

### Example Workflows

#### Uniform Application (Traditional)
- Weight Mode: `uniform`
- Block Weights: `1.0`
- Result: Standard LoRA application across all blocks

#### Targeted Enhancement
- Weight Mode: `block_specific`
- Block Range: `0-6`  
- Block Weights: `1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9`
- Result: Stronger influence on early blocks

#### Smooth Transition
- Weight Mode: `linear_interpolation`
- Interpolation Start: `1.5`
- Interpolation End: `0.5`
- Result: Gradual decrease from early to late blocks

## 🎛️ Node Parameters

### LoRA Block Weight Loader

#### Required Inputs
- **model**: The base model (Nunchaku/Flux/SD)
- **clip**: CLIP model
- **lora_name**: LoRA checkpoint file
- **strength_model**: Overall LoRA strength for model (−20.0 to 20.0)
- **strength_clip**: Overall LoRA strength for CLIP (−20.0 to 20.0)

#### Optional Inputs
- **block_weights**: Custom weight values (multiple formats supported)
- **weight_mode**: How weights are applied
  - `uniform`: Same weight for all blocks
  - `block_specific`: Custom weights per block
  - `linear_interpolation`: Linear gradient
  - `exponential`: Exponential curve
  - `gaussian`: Bell curve distribution
  - `custom_curve`: User-defined pattern
- **interpolation_start/end**: Start and end values for interpolation modes
- **interpolation_curve**: Curve factor for non-linear interpolations
- **block_range**: Which blocks to target (e.g., "all", "0-6", "7-12,15-18")
- **normalize_weights**: Normalize to maintain average strength
- **preserve_mean**: Keep mean value when normalizing
- **verbose**: Show detailed weight information
- **preset**: Use predefined weight patterns

### LoRA Block Weight Editor

A companion node for generating and visualizing weight patterns.

#### Parameters
- **total_blocks**: Number of blocks in your model
- **pattern**: Mathematical pattern to generate
- **amplitude**: Pattern amplitude
- **offset**: Base offset value
- **frequency**: Pattern frequency
- **phase**: Phase shift
- **custom_expression**: Mathematical expression for custom patterns

## 📊 Weight Input Formats

The node accepts multiple weight formats for maximum flexibility:

```python
# Single value (applies to all blocks)
"1.5"

# Space-separated
"1.0 1.2 1.4 1.2 1.0 0.8"

# Comma-separated
"1.0, 1.2, 1.4, 1.2, 1.0, 0.8"

# JSON array
"[1.0, 1.2, 1.4, 1.2, 1.0, 0.8]"

# JSON object with indices
'{"0": 1.5, "5": 1.2, "10": 0.8}'

# Multi-line block-specific (for Flux models)
"""
double_blocks: 1.0, 1.2, 1.4
single_blocks: 0.8, 0.9, 1.0
"""
```

## 🎨 Built-in Presets

- **linear_decay**: Gradual decrease from 1.5 to 0.5
- **linear_growth**: Gradual increase from 0.5 to 1.5
- **bell_curve**: Peak in middle blocks
- **u_shape**: Peak at extremes, trough in middle
- **emphasis_early**: Strong early blocks (first third)
- **emphasis_middle**: Strong middle blocks
- **emphasis_late**: Strong late blocks (last third)

## 🔧 Advanced Features

### Custom Presets

Create your own presets by adding JSON files to the `presets/` directory:

```json
{
  "name": "my_custom_preset",
  "description": "My custom weight distribution",
  "weights": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, ...]
}
```

For Flux-specific presets:
```json
{
  "name": "flux_custom",
  "description": "Custom Flux weights",
  "weights": {
    "double_blocks": [1.0, 1.1, 1.2, ...],
    "single_blocks": [0.9, 1.0, 1.1, ...]
  }
}
```

### Weight Visualization

The LoRA Block Weight Editor provides ASCII visualization:
```
Weight Distribution (blocks: 57)
Range: [0.500, 1.500]
Mean: 1.000, Std: 0.289

          ████████
       ███        ███
     ██              ██
   ██                  ██
 ██                      ██
─────────────────────────────
```

### Mathematical Expressions

Use custom expressions with the Weight Editor:
```python
# Sine wave
"1.0 + 0.5 * sin(2 * pi * i / n)"

# Exponential decay
"2.0 * exp(-i / (n / 3))"

# Step function
"1.5 if i < n/2 else 0.5"
```

## 🏗️ Architecture Support & Implementation

### Technical Implementation (ComfyUI 0.3.51+)

The node uses a sophisticated multi-layer approach:

1. **Weight Pre-Application**: Multiplies LoRA tensors by block-specific weights BEFORE loading
2. **Proper Key Mapping**: Uses `comfy.lora.model_lora_keys_unet()` and `model_lora_keys_clip()` for correct key generation
3. **Model Patching**: Applies patches via `model.add_patches()` with strength scaling
4. **Fallback System**: Three-level fallback ensures compatibility:
   - Primary: Weighted tensor loading with `comfy.lora.load_lora()`
   - Secondary: Path-based loading with adjusted strengths
   - Tertiary: Standard `comfy.sd.load_lora_for_models()` with averaged weights

### Flux Models
- **Architecture Detection**: Automatically identifies Flux models by checking for `double_blocks` and `single_blocks` attributes
- **Block Structure**: Properly handles 19 double blocks + 38 single blocks (57 total)
- **Weight Mapping**: Maps weights correctly with index offset for single blocks
- **Variant Support**: Validates against known Flux variants (19/22/24 double, 38/44/48 single)
- **Native Nunchaku Support**: Detects quantization config for optimized loading

### Stable Diffusion Models
- **Universal Compatibility**: Works with SD 1.5, SDXL, and variants
- **Block Detection**: Identifies input_blocks, output_blocks, middle_blocks
- **Pattern Matching**: Uses pre-compiled regex for efficient block identification
- **Graceful Degradation**: Falls back to uniform weights for unrecognized architectures

## 🎯 Use Cases

### Creative Control
- **Composition Control**: Strengthen early blocks for layout/pose
- **Style Transfer**: Adjust middle blocks for artistic style
- **Detail Enhancement**: Boost late blocks for fine details
- **Character Consistency**: Target specific blocks for facial features

### Technical Applications
- **LoRA Merging**: Different strengths for different LoRA aspects
- **Fine-tuning**: Selective layer updates
- **A/B Testing**: Compare different weight distributions
- **Research**: Analyze block contributions to generation

## 🐛 Troubleshooting

### LoRA Not Loading
- Ensure LoRA file is in `ComfyUI/models/loras/`
- Check console for specific error messages
- Verify ComfyUI version is 0.3.51 or higher
- Check that model has proper `.model` attribute
- Ensure CLIP has `.cond_stage_model` attribute

### API Compatibility Issues
- **"AttributeError: 'load_lora_for_models' not found"**: Update to ComfyUI 0.3.51+
- **"TypeError: load_lora() takes 2-3 args"**: Node is using correct API, check ComfyUI version
- **"Dictionary changed size during iteration"**: Fixed in latest version with proper dict comprehension

### Unexpected Results
- Enable verbose mode to see actual weights applied and architecture detected
- Check info output for block count validation
- Start with uniform weights as baseline
- Verify block range matches your model's architecture
- Check if normalization is affecting results (try with normalize_weights=False)

### Performance Issues
- Pre-compiled regex patterns minimize overhead (<1% vs standard loading)
- Large LoRA files (>2GB) will show memory warning
- CUDA memory cleanup runs automatically after loading
- Consider reducing block range for testing

### Memory Management
- Node includes automatic `torch.cuda.empty_cache()` for GPU memory
- File size checking warns for LoRAs over 2GB
- Efficient tensor operations minimize memory footprint

## 📈 Performance & Technical Details

### Performance Metrics
- **Memory**: Minimal additional memory usage with automatic CUDA cleanup
- **Speed**: < 1% overhead vs standard LoRA loading due to pre-compiled regex patterns
- **File Size Handling**: Automatic detection and warning for large LoRA files (>2GB)
- **Compatibility**: Works with all ComfyUI samplers and schedulers

### Key Implementation Features
- **Pre-compiled Regex Patterns**: All block matching patterns compiled at module load
- **Efficient Tensor Operations**: Direct multiplication without intermediate copies
- **Smart Fallback System**: Three-level fallback ensures loading success
- **Null Safety**: Comprehensive checks for model.model and clip.cond_stage_model
- **Memory Cleanup**: Automatic GPU memory management after large operations

### Code Quality
- **Production-Ready**: Extensive error handling and validation
- **Type Hints**: Full typing for better IDE support
- **Documentation**: Comprehensive docstrings and inline comments
- **Security**: Path traversal prevention in preset loading
- **Testing**: Validated across multiple ComfyUI versions and model types

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional mathematical presets
- Model architecture detection improvements
- Weight optimization algorithms
- Integration with other ComfyUI nodes

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- ComfyUI community for the framework
- Nunchaku and Flux model developers
- Contributors and testers

## 📚 References

- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI)
- [Flux Architecture](https://github.com/black-forest-labs/flux)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/bhvbhushan/ComfyUI-LoRABlockWeight/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bhvbhushan/ComfyUI-LoRABlockWeight/discussions)
- **Discord**: ComfyUI Discord #custom-nodes channel

---

**Note**: This node provides a general-purpose solution for per-block LoRA weight control. It works with any ComfyUI-compatible model that benefits from block-level weight control, including Flux, Nunchaku, SDXL, and SD 1.5 models.