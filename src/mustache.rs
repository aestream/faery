use pyo3::prelude::*;

#[pyclass]
#[derive(serde::Serialize, FromPyObject)]
pub struct Job {
    #[pyo3(get, set)]
    pub input: String,
    #[pyo3(get, set)]
    pub start: String,
    #[pyo3(get, set)]
    pub end: String,
    #[pyo3(get, set)]
    pub nickname: Option<String>,
}

#[pymethods]
impl Job {
    #[new]
    #[pyo3(signature = (input, start, end, nickname))]
    fn new(input: String, start: String, end: String, nickname: Option<String>) -> Self {
        Self {
            input,
            start,
            end,
            nickname,
        }
    }
}

#[derive(serde::Serialize)]
struct Data {
    jobs: Vec<Job>,
}

#[pyfunction]
pub fn render(template: &str, jobs: Vec<Job>) -> PyResult<String> {
    let template = mustache::compile_str(template)
        .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))?;
    let mut output = std::io::BufWriter::new(Vec::new());
    template
        .render(&mut output, &Data { jobs })
        .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))?;
    String::from_utf8(
        output
            .into_inner()
            .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))?,
    )
    .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))
}
