use std::sync::{Arc, Mutex};
use std::time::Instant;

pub fn convert_numpy_frame_to_rgba(
    frame: &numpy::ndarray::Array3<u8>,
) -> Result<(Vec<u8>, u32, u32), String> {
    let dimensions = frame.dim();
    let (height, width, channels) = dimensions;

    if channels != 3 && channels != 4 {
        return Err(format!(
            "Expected RGB or RGBA frame, got {} channels",
            channels
        ));
    }

    let frame_data = if channels == 4 {
        // Already RGBA, just copy the entire slice
        frame.as_slice().ok_or("Frame is not contiguous")?.to_vec()
    } else {
        // RGB to RGBA: add alpha channel for every 3 RGB bytes
        let rgb_slice = frame.as_slice().ok_or("Frame is not contiguous")?;
        let mut rgba_data = Vec::with_capacity(height * width * 4);

        for rgb_chunk in rgb_slice.chunks_exact(3) {
            rgba_data.extend_from_slice(rgb_chunk);
            rgba_data.push(255); // Add alpha
        }
        rgba_data
    };

    Ok((frame_data, width as u32, height as u32))
}

slint::slint! {
    import { Button, VerticalBox, HorizontalBox, GroupBox } from "std-widgets.slint";

    export component FrameViewerWindow inherits Window {
        title: "Faery Event Viewer";
        width: 800px;
        height: 600px;

        in property <image> current-frame;
        in property <string> frame-info: "";
        in property <bool> is-playing: false;

        callback play-pause();
        callback reset();
        callback next-frame();
        callback previous-frame();
        callback close-window();

        FocusScope {
            key-pressed(event) => {
                if (event.text == Key.Escape) {
                    close-window();
                    return accept;
                }
                return reject;
            }

            VerticalBox {
                padding: 10px;
                spacing: 10px;

            GroupBox {
                title: "Event Stream Visualization";
                Rectangle {
                    background: #1a1a1a;
                    border-radius: 4px;

                    Image {
                        source: current-frame;
                        image-fit: contain;
                        width: 100%;
                        height: 100%;
                    }
                }
            }

            Text {
                text: frame-info;
                font-size: 12px;
                color: #666;
            }

            HorizontalBox {
                spacing: 10px;
                alignment: center;

                Button {
                    text: "Previous";
                    clicked => { previous-frame(); }
                }

                Button {
                    text: is-playing ? "Pause" : "Play";
                    clicked => { play-pause(); }
                }

                Button {
                    text: "Next";
                    clicked => { next-frame(); }
                }

                Button {
                    text: "Reset";
                    clicked => { reset(); }
                }
            }
            }
        }
    }
}

// Communication object that can be shared between threads
#[derive(Clone)]
pub struct FrameStreamer {
    frame_count: Arc<Mutex<usize>>,
    shutdown: Arc<Mutex<bool>>,
    is_paused: Arc<Mutex<bool>>,
    frame_buffer: Arc<Mutex<Vec<numpy::ndarray::Array3<u8>>>>,
    current_frame_index: Arc<Mutex<usize>>,
    last_frame_time: Arc<Mutex<Instant>>,
    frame_rate: Arc<Mutex<f64>>,
    target_frame_rate: Arc<Mutex<Option<f64>>>,
    last_display_time: Arc<Mutex<Instant>>,
    stream_ended: Arc<Mutex<bool>>,
}

impl FrameStreamer {
    pub fn new(target_frame_rate: Option<f64>) -> Self {
        Self {
            frame_count: Arc::new(Mutex::new(0)),
            shutdown: Arc::new(Mutex::new(false)),
            is_paused: Arc::new(Mutex::new(false)),
            frame_buffer: Arc::new(Mutex::new(Vec::new())),
            current_frame_index: Arc::new(Mutex::new(0)),
            last_frame_time: Arc::new(Mutex::new(Instant::now())),
            frame_rate: Arc::new(Mutex::new(0.0)),
            target_frame_rate: Arc::new(Mutex::new(target_frame_rate)),
            last_display_time: Arc::new(Mutex::new(Instant::now())),
            stream_ended: Arc::new(Mutex::new(false)),
        }
    }

    pub fn is_shutdown(&self) -> bool {
        *self.shutdown.lock().unwrap()
    }

    pub fn set_shutdown(&self, shutdown: bool) {
        *self.shutdown.lock().unwrap() = shutdown;
    }

    pub fn is_stream_ended(&self) -> bool {
        *self.stream_ended.lock().unwrap()
    }

    pub fn set_stream_ended(&self, ended: bool) {
        *self.stream_ended.lock().unwrap() = ended;
    }

    pub fn add_frame(&self, frame: numpy::ndarray::Array3<u8>) {
        let mut buffer = self.frame_buffer.lock().unwrap();
        if !(self.is_paused() && buffer.len() >= 1024) {
            buffer.push(frame);
        }

        // Limit buffer size to 1024 frames
        if buffer.len() > 1024 {
            buffer.remove(0);
            // Adjust current index if needed
            let mut index = self.current_frame_index.lock().unwrap();
            if *index > 0 {
                *index -= 1;
            }
        }
        drop(buffer);

        self.increment_frame_count();
    }

    fn advance_frame(&self) {
        let buffer = self.frame_buffer.lock().unwrap();
        let mut index = self.current_frame_index.lock().unwrap();

        if !buffer.is_empty() {
            *index = buffer.len() - 1;
        }
    }

    pub fn is_paused(&self) -> bool {
        *self.is_paused.lock().unwrap()
    }

    pub fn toggle_pause(&self) -> bool {
        let mut is_paused = self.is_paused.lock().unwrap();
        *is_paused = !*is_paused;
        *is_paused
    }

    pub fn get_frame_count(&self) -> usize {
        *self.frame_count.lock().unwrap()
    }

    fn increment_frame_count(&self) -> usize {
        let mut count = self.frame_count.lock().unwrap();
        *count += 1;

        // Update frame rate calculation
        let now = Instant::now();
        let mut last_time = self.last_frame_time.lock().unwrap();
        let time_diff = now.duration_since(*last_time);

        if time_diff.as_millis() > 0 {
            // Ignore measurements after long pauses (> 500ms) to avoid artificially high rates
            if time_diff.as_millis() < 500 {
                let new_rate = 1000.0 / time_diff.as_millis() as f64;
                let mut frame_rate = self.frame_rate.lock().unwrap();
                // Use exponential moving average for smoother frame rate display
                *frame_rate = *frame_rate * 0.8 + new_rate * 0.2;
            }
        }
        *last_time = now;

        *count
    }

    pub fn reset(&self) {
        // Clear buffer except for one frame (if any exists)
        let mut buffer = self.frame_buffer.lock().unwrap();
        if !buffer.is_empty() {
            let last_frame = buffer.pop();
            buffer.clear();
            if let Some(frame) = last_frame {
                buffer.push(frame);
            }
        }

        // Reset frame index to 0 (will auto-adjust to half buffer as frames accumulate)
        drop(buffer);
        *self.current_frame_index.lock().unwrap() = 0;

        // Reset frame count and rate
        *self.frame_count.lock().unwrap() = 0;
        *self.frame_rate.lock().unwrap() = 0.0;

        // Resume playing if paused
        *self.is_paused.lock().unwrap() = false;
    }

    pub fn get_frame_rate(&self) -> f64 {
        *self.frame_rate.lock().unwrap()
    }

    pub fn should_display_frame(&self) -> bool {
        if let Some(target_rate) = *self.target_frame_rate.lock().unwrap() {
            let now = Instant::now();
            let last_display = *self.last_display_time.lock().unwrap();
            let min_interval = std::time::Duration::from_secs_f64(1.0 / target_rate);

            if now.duration_since(last_display) >= min_interval {
                *self.last_display_time.lock().unwrap() = now;
                true
            } else {
                false
            }
        } else {
            // No rate limit, always display
            true
        }
    }

    pub fn get_current_frame(&self) -> Option<numpy::ndarray::Array3<u8>> {
        if let Ok(buffer) = self.frame_buffer.try_lock() {
            if let Ok(index) = self.current_frame_index.try_lock() {
                if buffer.is_empty() {
                    None
                } else {
                    buffer.get(*index).cloned()
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn get_current_index(&self) -> usize {
        *self.current_frame_index.lock().unwrap()
    }

    pub fn next_frame(&self) -> Option<numpy::ndarray::Array3<u8>> {
        if !self.is_paused() {
            // Check if enough time has passed for next frame
            if self.should_display_frame() {
                self.advance_frame();
            }
            // If not enough time has passed, just return current frame
        } else {
            // Manual navigation when paused
            let buffer = self.frame_buffer.lock().unwrap();
            let mut index = self.current_frame_index.lock().unwrap();
            if !buffer.is_empty() {
                *index = (*index + 1).min(buffer.len() - 1);
            }
        }
        self.get_current_frame()
    }

    pub fn previous_frame(&self) -> Option<numpy::ndarray::Array3<u8>> {
        let buffer = self.frame_buffer.lock().unwrap();
        let mut index = self.current_frame_index.lock().unwrap();
        if !buffer.is_empty() {
            *index = if *index > 0 { *index - 1 } else { 0 };
        }
        drop(buffer);
        drop(index);
        self.get_current_frame()
    }
}

// Main FrameViewer that contains a window and uses a FrameStreamer for communication
pub struct FrameViewer {
    ui: FrameViewerWindow,
    streamer: FrameStreamer,
    _timer: slint::Timer, // Keep timer alive
}

impl FrameViewer {
    pub fn new(ui: FrameViewerWindow, streamer: FrameStreamer) -> Self {
        let timer = slint::Timer::default();
        let viewer = Self {
            ui,
            streamer,
            _timer: timer,
        };

        // Set initial playing state (not paused)
        viewer.ui.set_is_playing(true);

        viewer.setup_frame_handler();
        viewer.setup_callbacks();
        viewer
    }

    fn setup_frame_handler(&self) {
        let ui_weak = self.ui.as_weak();
        let streamer = self.streamer.clone();

        // Set timer frequency based on target frame rate, with 16ms (60fps) as minimum for UI responsiveness
        let timer_interval = if let Some(target_rate) = *streamer.target_frame_rate.lock().unwrap() {
            let target_interval = 1000.0 / target_rate;
            std::cmp::max(16, target_interval as u64)
        } else {
            16 // Default to 60fps for maximum responsiveness when no rate limit
        };

        self._timer.start(
            slint::TimerMode::Repeated,
            std::time::Duration::from_millis(timer_interval),
            move || {
                // Check if we should shutdown
                if streamer.is_shutdown() {
                    return;
                }

                if !streamer.is_paused() {
                    streamer.next_frame();
                }
                if let Some(frame) = streamer.get_current_frame() {
                    if let Ok((frame_data, width, height)) = convert_numpy_frame_to_rgba(&frame) {
                        if let Ok(buffer) = streamer.frame_buffer.try_lock() {
                            if let Ok(index) = streamer.current_frame_index.try_lock() {
                                let buffer_len = buffer.len();
                                let current_index = *index;
                                drop(buffer);
                                drop(index);

                                let frame_rate = streamer.get_frame_rate();
                                let stream_ended = streamer.is_stream_ended();
                                ui_weak
                                    .upgrade_in_event_loop(move |ui| {
                                        let image = slint::Image::from_rgba8(
                                            slint::SharedPixelBuffer::clone_from_slice(
                                                &frame_data,
                                                width,
                                                height,
                                            ),
                                        );
                                        ui.set_current_frame(image);
                                        let status = if stream_ended {
                                            format!(
                                                "Frame buffer: {}/{} - {:.1} Hz (Stream ended)",
                                                current_index + 1,
                                                buffer_len,
                                                frame_rate
                                            )
                                        } else {
                                            format!(
                                                "Frame buffer: {}/{} - {:.1} Hz",
                                                current_index + 1,
                                                buffer_len,
                                                frame_rate
                                            )
                                        };
                                        ui.set_frame_info(status.into());
                                    })
                                    .unwrap_or_else(|_| {
                                        println!(
                                            "UI failed to render (reference could not be upgraded)"
                                        );
                                    });
                            }
                        }
                    } else {
                        println!("Failed to convert frame to RGBA");
                    }
                }
            },
        );
    }

    fn setup_callbacks(&self) {
        // Set up close callback
        let streamer_clone = self.streamer.clone();
        self.ui.window().on_close_requested(move || {
            streamer_clone.set_shutdown(true);
            slint::CloseRequestResponse::HideWindow
        });

        // Set up callback for closing the window
        self.ui.on_close_window({
            let streamer = self.streamer.clone();
            move || {
                streamer.set_shutdown(true);
            }
        });

        self.ui.on_play_pause({
            let streamer = self.streamer.clone();
            let ui_weak = self.ui.as_weak();
            move || {
                let is_paused = streamer.toggle_pause();

                // Update the UI button text
                ui_weak
                    .upgrade_in_event_loop(move |ui| {
                        ui.set_is_playing(!is_paused);
                    })
                    .unwrap_or_else(|_| {
                        println!("Failed to update play/pause button state");
                    });
            }
        });

        self.ui.on_reset({
            let streamer = self.streamer.clone();
            let ui_weak = self.ui.as_weak();
            move || {
                streamer.reset();

                // Update UI to reflect that we're playing again
                ui_weak
                    .upgrade_in_event_loop(move |ui| {
                        ui.set_is_playing(true);
                    })
                    .unwrap_or_else(|_| {
                        println!("Failed to update play state after reset");
                    });
            }
        });

        self.ui.on_next_frame({
            let streamer = self.streamer.clone();
            move || {
                streamer.next_frame();
            }
        });

        self.ui.on_previous_frame({
            let streamer = self.streamer.clone();
            move || {
                streamer.previous_frame();
            }
        });
    }

    pub fn is_shutdown(&self) -> bool {
        self.streamer.is_shutdown()
    }

    pub fn get_streamer(&self) -> FrameStreamer {
        self.streamer.clone()
    }

    pub fn run(&self) -> Result<(), slint::PlatformError> {
        self.ui.run()
    }

    pub fn show(&self) -> Result<(), slint::PlatformError> {
        self.ui.show()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_viewer_with_mock_frames() {
        // Create mock RGB frames
        let frame1 = numpy::ndarray::Array3::<u8>::zeros((100, 100, 3));
        let mut frame2 = numpy::ndarray::Array3::<u8>::zeros((100, 100, 3));

        // Add some red pixels to frame2
        for i in 40..60 {
            for j in 40..60 {
                frame2[[i, j, 0]] = 255; // Red channel
            }
        }

        // Test conversion function without GUI initialization
        let (rgba_data, width, height) = convert_numpy_frame_to_rgba(&frame2).unwrap();

        // Verify dimensions
        assert_eq!(width, 100);
        assert_eq!(height, 100);
        assert_eq!(rgba_data.len(), 100 * 100 * 4);

        // Verify red pixels are correctly converted
        let red_pixel_idx = (50 * 100 + 50) * 4; // Middle of red square
        assert_eq!(rgba_data[red_pixel_idx], 255); // R
        assert_eq!(rgba_data[red_pixel_idx + 1], 0); // G
        assert_eq!(rgba_data[red_pixel_idx + 2], 0); // B
        assert_eq!(rgba_data[red_pixel_idx + 3], 255); // A
    }

    #[test]
    fn test_frame_viewer_index_add() {
        let streamer = FrameStreamer::new(None);
        let frame = numpy::ndarray::Array3::<u8>::zeros((100, 100, 3));
        let frame1 = frame.clone();
        assert_eq!(streamer.get_frame_count(), 0);
        streamer.add_frame(frame);
        assert_eq!(streamer.get_frame_count(), 1);
        streamer.add_frame(frame1);
        assert_eq!(streamer.get_frame_count(), 2);

        assert_eq!(streamer.get_current_index(), 0);
        streamer.next_frame();
        assert_eq!(streamer.get_frame_count(), 2);
        assert_eq!(streamer.get_current_index(), 1);
        streamer.next_frame();
        assert_eq!(streamer.get_current_index(), 1); // We expect one because len == 2
    }
}
