use pyo3::prelude::*;

mod aedat;
mod csv;
mod dat;
mod event_stream;
mod evt;
mod types;
mod utilities;

#[pymodule]
fn faery(python: Python<'_>, module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    {
        let submodule = PyModule::new_bound(python, "aedat")?;
        submodule.add_class::<aedat::Decoder>()?;
        submodule.add_class::<aedat::Encoder>()?;
        submodule.add_class::<aedat::Frame>()?;
        submodule.add_class::<aedat::Track>()?;
        submodule.add("LZ4_FASTEST", ("lz4", utilities::LZ4_MINIMUM_LEVEL))?;
        submodule.add("LZ4_DEFAULT", ("lz4", utilities::LZ4_DEFAULT_LEVEL))?;
        submodule.add("LZ4_HIGHEST", ("lz4", utilities::LZ4_MAXIMUM_LEVEL))?;
        submodule.add("ZSTD_FASTEST", ("zstd", utilities::ZSTD_MINIMUM_LEVEL))?;
        submodule.add("ZSTD_DEFAULT", ("zstd", utilities::ZSTD_DEFAULT_LEVEL))?;
        submodule.add("ZSTD_HIGHEST", ("zstd", utilities::ZSTD_MAXIMUM_LEVEL))?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new_bound(python, "csv")?;
        submodule.add_class::<csv::Decoder>()?;
        submodule.add_class::<csv::Encoder>()?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new_bound(python, "dat")?;
        submodule.add_class::<dat::Decoder>()?;
        submodule.add_class::<dat::Encoder>()?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new_bound(python, "event_stream")?;
        submodule.add_class::<event_stream::Decoder>()?;
        submodule.add_class::<event_stream::Encoder>()?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new_bound(python, "evt")?;
        submodule.add_class::<evt::Decoder>()?;
        submodule.add_class::<evt::Encoder>()?;
        module.add_submodule(&submodule)?;
    }
    Ok(())
}
