use std::thread;
use pyo3::prelude::*;
use numpy::PyArrayMethods;

pub mod viewer;

pub use viewer::{FrameViewer, FrameViewerWindow, FrameStreamer};

#[pyfunction]
pub fn run_frame_viewer_from_iterator(py: Python, frame_stream: PyObject) -> PyResult<()> {
    // Create window first to catch any errors directly
    let ui = FrameViewerWindow::new()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create window: {}", e)))?;

    // Create streamer for thread-safe communication
    let streamer = FrameStreamer::new();

    // Create viewer with window and streamer
    let viewer = FrameViewer::new(ui, streamer.clone());

    // Show the window
    viewer.show().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to show window: {}", e)))?;

    // Background thread to collect frames from Python iterator
    let iter = frame_stream.call_method0(py, "__iter__")?;
    let streamer_clone = streamer.clone();
    let _handle = thread::spawn(move || {

        loop {
            // Check for shutdown signal
            if streamer_clone.is_shutdown() {
                break;
            }

            let result = Python::with_gil(|py| {
               iter.call_method0(py, "__next__")
            });

            match result {
                Ok(frame_obj) => {
                    if let Ok(frame_data) = Python::with_gil(|py| -> PyResult<numpy::ndarray::Array3<u8>> {
                        let pixels = frame_obj.getattr(py, "pixels")?;
                        let frame_array = pixels.bind(py).downcast::<numpy::PyArray3<u8>>()?;
                        let readonly_frame = frame_array.readonly();
                        let array = readonly_frame.as_array();
                        Ok(array.to_owned())
                    }) {
                        streamer_clone.add_frame(frame_data);
                    }
                }
                Err(_) => {
                    streamer_clone.set_shutdown(true);
                    break;
                }
            }
        }
    });

    // Timer to check for shutdown and quit event loop
    let streamer_check = streamer.clone();
    let timer = slint::Timer::default();
    timer.start(slint::TimerMode::Repeated, std::time::Duration::from_millis(100), move || {
        if streamer_check.is_shutdown() {
            println!("Shutdown detected, quitting event loop");
            let _ = slint::quit_event_loop();
        }
    });

    // Start the GUI in a thread so we can return and release the GIL
    py.allow_threads(|| {
        // Run event loop on main thread
        slint::run_event_loop()
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to run event loop: {}", e)))?;

    Ok(())
}
