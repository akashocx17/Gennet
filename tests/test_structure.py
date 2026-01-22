"""Simple structure test that doesn't require heavy dependencies."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    # Test config imports
    try:
        from gennet.configs.model_config import (
            ModelConfig,
            ModernBERTConfig,
            Siglip2Config,
            FusionConfig,
            RLConfig
        )
        print("✓ Config modules imported successfully")
        
        # Test config creation
        config = ModelConfig()
        assert config.text_config is not None
        assert config.vision_config is not None
        assert config.fusion_config is not None
        assert config.rl_config is not None
        print("✓ Config objects created successfully")
        
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    return True


def test_structure():
    """Test directory structure."""
    print("\nTesting directory structure...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    required_dirs = [
        'gennet',
        'gennet/models',
        'gennet/configs',
        'gennet/training',
        'gennet/utils',
        'examples'
    ]
    
    required_files = [
        'gennet/__init__.py',
        'gennet/models/__init__.py',
        'gennet/models/text_encoder.py',
        'gennet/models/vision_encoder.py',
        'gennet/models/fusion_layer.py',
        'gennet/models/rl_layer.py',
        'gennet/models/multimodal_model.py',
        'gennet/configs/__init__.py',
        'gennet/configs/model_config.py',
        'gennet/training/__init__.py',
        'gennet/training/trainer.py',
        'gennet/utils/__init__.py',
        'gennet/utils/data_utils.py',
        'examples/train_example.py',
        'requirements.txt',
        'setup.py',
        'README.md'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.isdir(full_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_good = False
    
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.isfile(full_path):
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ File missing: {file_path}")
            all_good = False
    
    return all_good


def test_code_structure():
    """Test code structure and key classes."""
    print("\nTesting code structure...")
    
    try:
        # These imports should work even without torch/transformers
        # since we're just checking the structure exists
        with open('gennet/models/text_encoder.py', 'r') as f:
            content = f.read()
            assert 'class ModernBERTEncoder' in content
            assert 'class DAPPooling' in content
            print("✓ ModernBERTEncoder class defined")
        
        with open('gennet/models/vision_encoder.py', 'r') as f:
            content = f.read()
            assert 'class Siglip2VisionEncoder' in content
            print("✓ Siglip2VisionEncoder class defined")
        
        with open('gennet/models/fusion_layer.py', 'r') as f:
            content = f.read()
            assert 'class CrossModalFusion' in content
            print("✓ CrossModalFusion class defined")
        
        with open('gennet/models/rl_layer.py', 'r') as f:
            content = f.read()
            assert 'class ReinforcementLearningLayer' in content
            assert 'class PolicyNetwork' in content
            assert 'class ValueNetwork' in content
            print("✓ RL components defined")
        
        with open('gennet/models/multimodal_model.py', 'r') as f:
            content = f.read()
            assert 'class MultiModalModel' in content
            print("✓ MultiModalModel class defined")
        
        with open('gennet/training/trainer.py', 'r') as f:
            content = f.read()
            assert 'class MultiModalTrainer' in content
            print("✓ MultiModalTrainer class defined")
        
        return True
        
    except Exception as e:
        print(f"✗ Code structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Gennet Structure Test")
    print("=" * 60)
    
    results = []
    
    # Test structure
    results.append(("Directory Structure", test_structure()))
    
    # Test code structure
    results.append(("Code Structure", test_code_structure()))
    
    # Test imports (config only, doesn't need torch)
    results.append(("Config Imports", test_imports()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
