# Contributing

Merci de considérer une contribution à Watermark Evil ! Ce document fournit des lignes directrices pour contribuer au projet.

## 📋 Table des Matières

- [Code de Conduite](#code-de-conduite)
- [Comment Contribuer](#comment-contribuer)
- [Style de Code](#style-de-code)
- [Tests](#tests)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code de Conduite

Ce projet suit un Code de Conduite qui définit les standards de comportement pour tous les participants. En participant, vous acceptez de respecter ces règles.

### Nos Standards

- Utiliser un langage accueillant et inclusif
- Respecter les différents points de vue
- Accepter les critiques constructives
- Se concentrer sur ce qui est le mieux pour la communauté
- Faire preuve d'empathie envers les autres membres

## Comment Contribuer

### 1. Préparer l'Environnement

```bash
# Fork et clone
git clone https://github.com/your-username/watermark-evil.git
cd watermark-evil

# Créer une branche
git checkout -b feature/amazing-feature

# Installer les dépendances
pip install -r requirements.txt
cargo build
```

### 2. Développer

1. **Choisir une Issue**
   - Vérifier les issues existantes
   - Discuter des changements majeurs dans une issue
   - Assigner l'issue à vous-même

2. **Écrire le Code**
   - Suivre le style de code
   - Ajouter des tests
   - Mettre à jour la documentation

3. **Commit**
   ```bash
   git add .
   git commit -m "feat: Add amazing feature"
   ```

### 3. Soumettre

1. **Push les Changements**
   ```bash
   git push origin feature/amazing-feature
   ```

2. **Créer une Pull Request**
   - Utiliser le template
   - Lier les issues pertinentes
   - Décrire les changements

## Style de Code

### Python

```python
# Correct
def process_image(
    image: np.ndarray,
    quality: float = 0.8
) -> np.ndarray:
    """Process an image.
    
    Args:
        image: Input image
        quality: Quality factor
        
    Returns:
        Processed image
        
    Raises:
        ValueError: If quality is invalid
    """
    if not 0 <= quality <= 1:
        raise ValueError("Quality must be between 0 and 1")
        
    return image

# Incorrect
def process_image(image,quality = 0.8):
    return image
```

### Rust

```rust
// Correct
pub struct ImageProcessor {
    config: ProcessorConfig,
    memory_pool: GpuMemoryPool,
}

impl ImageProcessor {
    pub fn new(config: ProcessorConfig) -> Result<Self> {
        Ok(Self {
            config,
            memory_pool: GpuMemoryPool::new()?,
        })
    }
}

// Incorrect
struct imageProcessor {
    Config: ProcessorConfig,
    memory_pool: GpuMemoryPool
}
```

## Tests

### Écrire des Tests

```python
# test_detector.py
def test_detector_validation():
    """Test detector input validation."""
    detector = WatermarkDetector()
    
    # Test valid input
    result = detector.validate_input(valid_image)
    assert result.is_ok()
    
    # Test invalid input
    with pytest.raises(ValidationError):
        detector.validate_input(invalid_image)
```

### Exécuter les Tests

```bash
# Tests Python
pytest tests/ --cov=src

# Tests Rust
cargo test --release

# Tests d'intégration
python -m pytest tests/integration/
```

## Documentation

### Docstrings

```python
def remove_watermark(
    image_path: str,
    output_path: str,
    quality: float = 0.8
) -> bool:
    """Remove watermark from image.
    
    This function detects and removes watermarks from the input image
    using our ML models and reconstruction engine.
    
    Args:
        image_path: Path to input image
        output_path: Path to save result
        quality: Quality factor (0-1)
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        ValueError: If paths are invalid
        ProcessingError: If removal fails
        
    Example:
        >>> remove_watermark("input.jpg", "output.jpg", 0.9)
        True
    """
```

### Commentaires

```rust
/// Configuration for the processing engine
pub struct ProcessorConfig {
    /// Maximum image size in pixels
    pub max_size: u32,
    
    /// Quality factor (0-1)
    pub quality: f32,
    
    /// Enable CUDA acceleration
    pub cuda_enabled: bool,
}
```

## Release Process

### 1. Préparation

1. Mettre à jour la version :
   ```bash
   bump2version patch  # ou minor/major
   ```

2. Mettre à jour CHANGELOG.md :
   ```markdown
   ## [1.1.0] - 2025-01-13
   
   ### Added
   - Nouvelle fonctionnalité X
   - Support pour Y
   
   ### Fixed
   - Bug Z
   ```

### 2. Tests

```bash
# Exécuter tous les tests
pytest tests/
cargo test --release
python -m pytest tests/integration/

# Vérifier la couverture
pytest --cov=src --cov-report=html
```

### 3. Build

```bash
# Python
python setup.py sdist bdist_wheel

# Rust
cargo build --release
```

### 4. Publication

1. Tag Git :
   ```bash
   git tag -a v1.1.0 -m "Version 1.1.0"
   git push origin v1.1.0
   ```

2. Publier :
   ```bash
   # Python
   twine upload dist/*
   
   # Cargo
   cargo publish
   ```

## Questions ?

Pour toute question :
1. Vérifier la documentation
2. Chercher dans les issues
3. Créer une nouvelle issue
4. Contacter les mainteneurs

Merci de contribuer ! 🎉
