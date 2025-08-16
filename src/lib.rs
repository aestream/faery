use pyo3::prelude::*;

mod aedat;
mod csv;
mod dat;
mod es;
mod evt;
mod font;
mod gif;
mod image;
mod job_metadata;
mod mp4;
mod mustache;
mod raster;
mod render;
//mod event_spectrogram;
mod types;
mod utilities;

#[pymodule]
#[pyo3(name = "extension")]
fn faery(python: Python<'_>, module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    {
        let submodule = PyModule::new(python, "aedat")?;
        submodule.add_class::<aedat::Decoder>()?;
        submodule.add_class::<aedat::Encoder>()?;
        submodule.add_class::<aedat::Frame>()?;
        submodule.add_class::<aedat::Track>()?;
        submodule.add_class::<aedat::DescriptionAttribute>()?;
        submodule.add_class::<aedat::DescriptionNode>()?;
        submodule.add("LZ4_FASTEST", ("lz4", utilities::LZ4_MINIMUM_LEVEL))?;
        submodule.add("LZ4_DEFAULT", ("lz4", utilities::LZ4_DEFAULT_LEVEL))?;
        submodule.add("LZ4_HIGHEST", ("lz4", utilities::LZ4_MAXIMUM_LEVEL))?;
        submodule.add("ZSTD_FASTEST", ("zstd", utilities::ZSTD_MINIMUM_LEVEL))?;
        submodule.add("ZSTD_DEFAULT", ("zstd", utilities::ZSTD_DEFAULT_LEVEL))?;
        submodule.add("ZSTD_HIGHEST", ("zstd", utilities::ZSTD_MAXIMUM_LEVEL))?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new(python, "csv")?;
        submodule.add_class::<csv::Decoder>()?;
        submodule.add_class::<csv::Encoder>()?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new(python, "dat")?;
        submodule.add_class::<dat::Decoder>()?;
        submodule.add_class::<dat::Encoder>()?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new(python, "es")?;
        submodule.add_class::<es::Decoder>()?;
        submodule.add_class::<es::Encoder>()?;
        module.add_submodule(&submodule)?;
    }
    //{
    //    let submodule = PyModule::new(python, "event_spectrogram")?;
    //    submodule.add_class::<event_spectrogram::EventSpectrogram>()?;
    //    module.add_submodule(&submodule)?;
    //}
    {
        let submodule = PyModule::new(python, "evt")?;
        submodule.add_class::<evt::Decoder>()?;
        submodule.add_class::<evt::Encoder>()?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new(python, "gif")?;
        submodule.add_class::<gif::Encoder>()?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new(python, "image")?;
        submodule.add_function(wrap_pyfunction!(image::decode, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(image::encode, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(image::annotate, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(image::resize, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(image::overlay, &submodule)?)?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new(python, "job_metadata")?;
        submodule.add_class::<job_metadata::Task>()?;
        submodule.add_function(wrap_pyfunction!(job_metadata::read, &submodule)?)?;
        submodule.add_function(wrap_pyfunction!(job_metadata::write, &submodule)?)?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new(python, "mp4")?;
        submodule.add_class::<mp4::Encoder>()?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new(python, "mustache")?;
        submodule.add_class::<mustache::Job>()?;
        submodule.add_function(wrap_pyfunction!(mustache::render, &submodule)?)?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new(python, "raster")?;
        submodule.add_function(wrap_pyfunction!(raster::render, &submodule)?)?;
        module.add_submodule(&submodule)?;
    }
    {
        let submodule = PyModule::new(python, "render")?;
        submodule.add_class::<render::Renderer>()?;
        module.add_submodule(&submodule)?;
    }
    Ok(())
}
