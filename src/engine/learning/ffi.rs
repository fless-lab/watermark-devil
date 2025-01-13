use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use anyhow::Result;
use std::sync::Arc;

use super::trainer::ModelTrainer;
use super::config::LearningConfig;
use super::{PatternType, TrainingData};

/// Wrapper Python pour ModelTrainer
#[pyclass]
pub struct PyModelTrainer {
    inner: Arc<ModelTrainer>,
}

#[pymethods]
impl PyModelTrainer {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(ModelTrainer::new()),
        })
    }

    /// Train models with provided data
    #[pyo3(text_signature = "($self, data, /)")]
    fn train_models(&self, py: Python<'_>, data: &PyDict) -> PyResult<()> {
        let training_data = convert_py_data_to_rust(py, data)?;
        self.inner.train_models(training_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get current model metrics
    #[pyo3(text_signature = "($self, /)")]
    fn get_metrics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let metrics = self.inner.get_metrics()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let dict = PyDict::new(py);
        dict.set_item("accuracy", metrics.accuracy)?;
        dict.set_item("precision", metrics.precision)?;
        dict.set_item("recall", metrics.recall)?;
        dict.set_item("f1_score", metrics.f1_score)?;
        
        Ok(dict.into())
    }

    /// Check if training is needed
    #[pyo3(text_signature = "($self, /)")]
    fn should_train(&self) -> PyResult<bool> {
        self.inner.should_train()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

/// Convert Python training data to Rust format
fn convert_py_data_to_rust(py: Python<'_>, data: &PyDict) -> PyResult<TrainingData> {
    let features = data.get_item("features")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing features"))?;
    let labels = data.get_item("labels")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing labels"))?;
    
    let features: Vec<f32> = features.extract()?;
    let labels: Vec<String> = labels.extract()?;
    
    let pattern_types: Vec<PatternType> = labels.into_iter()
        .map(|s| match s.as_str() {
            "logo" => PatternType::Logo,
            "text" => PatternType::Text,
            "repetitive" => PatternType::Repetitive,
            "complex" => PatternType::Complex,
            _ => PatternType::Unknown,
        })
        .collect();
    
    Ok(TrainingData {
        features,
        labels: pattern_types,
    })
}

/// Python module initialization
#[pymodule]
fn watermark_evil_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyModelTrainer>()?;
    Ok(())
}
