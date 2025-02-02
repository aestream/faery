use pyo3::prelude::*;

use crate::types;

#[pyclass]
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct Task {
    #[pyo3(get, set)]
    pub task_hash: String,
    #[pyo3(get, set)]
    pub task_code: String,
}

#[pymethods]
impl Task {
    #[new]
    fn new(task_hash: String, task_code: String) -> Self {
        Self {
            task_hash,
            task_code,
        }
    }
}

pub type JobMetadata = std::collections::BTreeMap<String, Task>;

#[pyfunction]
pub fn read(path: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<JobMetadata> {
    let contents = std::fs::read_to_string(types::python_path_to_string(path)?)
        .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))?;
    toml::from_str(&contents)
        .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))
}

#[pyfunction]
pub fn write(
    job_metadata: JobMetadata,
    path: &pyo3::Bound<'_, pyo3::types::PyAny>,
) -> PyResult<()> {
    let contents = toml::to_string(&job_metadata)
        .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))?;
    std::fs::write(types::python_path_to_string(path)?, contents)
        .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))
}
