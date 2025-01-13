use std::path::Path;
use anyhow::Result;
use image::{DynamicImage, ImageBuffer};
use opencv::core::{Mat, MatTraitConst};
use pyo3::prelude::*;
use numpy::{PyArray, PyArray3};

pub fn mat_to_tensor(mat: &Mat) -> Result<PyObject> {
    Python::with_gil(|py| {
        let torch = PyModule::import(py, "torch")?;
        let numpy = PyModule::import(py, "numpy")?;
        
        // Convert Mat to NumPy array
        let array = numpy.getattr("array")?.call1((mat.data_bytes()?,))?;
        
        // Convert to torch tensor
        let tensor = torch.getattr("from_numpy")?.call1((array,))?;
        
        // Add batch dimension and normalize
        let tensor = tensor.call_method1("unsqueeze", (0,))?;
        let tensor = tensor.call_method1("float", ())?;
        let tensor = tensor.call_method1("div", (255.0,))?;
        
        Ok(tensor.into())
    })
}

pub fn tensor_to_mat(tensor: &PyObject) -> Result<Mat> {
    Python::with_gil(|py| {
        // Convert to NumPy array
        let array = tensor.call_method0(py, "numpy")?;
        let array: &PyArray3<u8> = array.extract()?;
        
        // Create Mat from array data
        let height = array.shape()[0] as i32;
        let width = array.shape()[1] as i32;
        let channels = array.shape()[2] as i32;
        
        let mut mat = unsafe {
            Mat::new_rows_cols_with_data(
                height,
                width,
                opencv::core::CV_8UC(channels),
                array.as_raw_array_ptr() as *mut _,
                Mat::AUTO_STEP,
            )?
        };
        
        Ok(mat)
    })
}

pub fn save_model(model: &PyObject, path: &Path) -> Result<()> {
    Python::with_gil(|py| {
        let torch = PyModule::import(py, "torch")?;
        let state_dict = model.call_method0(py, "state_dict")?;
        torch.getattr("save")?.call1((state_dict, path.to_str().unwrap()))?;
        Ok(())
    })
}

pub fn load_model(path: &Path, model_class: &str) -> Result<PyObject> {
    Python::with_gil(|py| {
        let torch = PyModule::import(py, "torch")?;
        
        // Create model instance
        let model = match model_class {
            "LogoDetector" => {
                PyModule::import(py, "ml.models.logo_detector.model")?
                    .getattr("LogoDetector")?
                    .call0()?
            }
            "TextDetector" => {
                PyModule::import(py, "ml.models.text_detector.model")?
                    .getattr("TextDetector")?
                    .call0()?
            }
            "PatternDetector" => {
                PyModule::import(py, "ml.models.pattern_detector.model")?
                    .getattr("PatternDetector")?
                    .call0()?
            }
            "TransparencyDetector" => {
                PyModule::import(py, "ml.models.transparency_detector.model")?
                    .getattr("TransparencyDetector")?
                    .call0()?
            }
            _ => return Err(anyhow::anyhow!("Unknown model class: {}", model_class)),
        };
        
        // Load state dict
        let state_dict = torch.getattr("load")?.call1((path.to_str().unwrap(),))?;
        model.call_method1("load_state_dict", (state_dict,))?;
        
        Ok(model.into())
    })
}

pub fn get_device() -> Result<String> {
    Python::with_gil(|py| {
        let torch = PyModule::import(py, "torch")?;
        let cuda_available: bool = torch
            .getattr("cuda")?
            .getattr("is_available")?
            .call0()?
            .extract()?;
            
        Ok(if cuda_available {
            "cuda".to_string()
        } else {
            "cpu".to_string()
        })
    })
}

pub fn to_device(tensor: &PyObject, device: &str) -> Result<PyObject> {
    Python::with_gil(|py| {
        Ok(tensor.call_method1(py, "to", (device,))?.into())
    })
}

pub fn apply_transforms(image: &Mat) -> Result<PyObject> {
    Python::with_gil(|py| {
        let transforms = PyModule::import(py, "torchvision.transforms")?;
        let transform = transforms.call_method1(
            "Compose",
            (vec![
                transforms.call_method1("Resize", ((224, 224),))?,
                transforms.call_method1("ToTensor", ())?,
                transforms.call_method1(
                    "Normalize",
                    (
                        vec![0.485, 0.456, 0.406],
                        vec![0.229, 0.224, 0.225],
                    ),
                )?,
            ],),
        )?;
        
        let tensor = mat_to_tensor(image)?;
        Ok(transform.call1((tensor,))?.into())
    })
}

pub fn create_dataloader(
    tensors: Vec<PyObject>,
    labels: Vec<i64>,
    batch_size: usize,
    shuffle: bool,
) -> Result<PyObject> {
    Python::with_gil(|py| {
        let torch = PyModule::import(py, "torch")?;
        let utils = torch.getattr("utils")?;
        let data = utils.getattr("data")?;
        
        let dataset = data.getattr("TensorDataset")?.call1((
            torch.getattr("stack")?.call1((tensors,))?,
            torch.getattr("tensor")?.call1((labels,))?,
        ))?;
        
        let dataloader = data.getattr("DataLoader")?.call1((
            dataset,
            vec![("batch_size", batch_size), ("shuffle", shuffle)],
        ))?;
        
        Ok(dataloader.into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::Scalar;
    use tempfile::tempdir;
    
    #[test]
    fn test_tensor_conversion() -> Result<()> {
        let mat = Mat::new_rows_cols_with_default(
            100,
            100,
            opencv::core::CV_8UC3,
            Scalar::all(255.0),
        )?;
        
        let tensor = mat_to_tensor(&mat)?;
        let mat2 = tensor_to_mat(&tensor)?;
        
        assert_eq!(mat.size()?, mat2.size()?);
        assert_eq!(mat.typ(), mat2.typ());
        
        Ok(())
    }
    
    #[test]
    fn test_model_save_load() -> Result<()> {
        let temp_dir = tempdir()?;
        let model_path = temp_dir.path().join("model.pt");
        
        Python::with_gil(|py| {
            let model = PyModule::import(py, "ml.models.logo_detector.model")?
                .getattr("LogoDetector")?
                .call0()?;
                
            save_model(&model.into(), &model_path)?;
            
            let loaded_model = load_model(&model_path, "LogoDetector")?;
            assert!(loaded_model.call_method0(py, "parameters")?.is_some());
            
            Ok(())
        })
    }
    
    #[test]
    fn test_device_detection() -> Result<()> {
        let device = get_device()?;
        assert!(device == "cuda" || device == "cpu");
        Ok(())
    }
    
    #[test]
    fn test_transforms() -> Result<()> {
        let mat = Mat::new_rows_cols_with_default(
            100,
            100,
            opencv::core::CV_8UC3,
            Scalar::all(255.0),
        )?;
        
        let transformed = apply_transforms(&mat)?;
        
        Python::with_gil(|py| {
            let shape = transformed.call_method0(py, "shape")?;
            assert_eq!(shape.extract::<(i64, i64, i64)>()?, (3, 224, 224));
            Ok(())
        })
    }
}
