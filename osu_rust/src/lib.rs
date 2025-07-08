use pyo3::prelude::*;
use pyo3::types::PyDict;

mod beatmap;
mod replay;
mod types;

pub use types::*;

/// Fast beatmap parsing and frame generation
#[pyfunction]
fn parse_beatmap_fast(file_path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let beatmap = beatmap::parse_beatmap(&file_path)?;
        let py_dict = PyDict::new(py);
        
        // Convert Rust beatmap to Python dict
        beatmap.to_py_dict(py_dict)?;
        
        Ok(py_dict.into())
    })
}

/// Fast replay parsing
#[pyfunction]
fn parse_replay_fast(file_path: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let replay = replay::parse_replay(&file_path)?;
        let py_dict = PyDict::new(py);
        
        // Convert Rust replay to Python dict
        replay.to_py_dict(py_dict)?;
        
        Ok(py_dict.into())
    })
}

/// Generate beatmap frames exactly like Python
#[pyfunction]
fn generate_beatmap_frames_exact(
    file_path: String,
    start_time: i32,
    end_time: i32,
    frame_rate: i32,
    screen_width: f32,
    screen_height: f32,
) -> PyResult<Vec<Vec<f32>>> {
    let mut beatmap = beatmap::parse_beatmap(&file_path)?;
    let frames = beatmap::generate_beatmap_frames_exact(&mut beatmap, start_time, end_time, frame_rate, screen_width, screen_height);
    Ok(frames)
}

/// Generate replay frames exactly like Python
#[pyfunction]
fn generate_replay_frames_exact(
    file_path: String,
    start_time: i32,
    end_time: i32,
    frame_rate: i32,
    screen_width: f32,
    screen_height: f32,
) -> PyResult<Vec<Vec<f32>>> {
    // TODO this has to reparse the replay - which is arguably the most expenisve thing
    // to parse. we shoudl cache loaded replays somewhere
    let replay = replay::parse_replay(&file_path)?;
    let frames = replay::generate_replay_frames_exact(&replay, start_time, end_time, frame_rate, screen_width, screen_height);
    Ok(frames)
}

/// Batch process multiple files sequentially
#[pyfunction]
fn batch_parse_files(file_pairs: Vec<(String, String)>) -> PyResult<Vec<PyObject>> {
    Python::with_gil(|py| {
        let mut results = Vec::new();
        
        for (replay_path, beatmap_path) in file_pairs {
            let replay = replay::parse_replay(&replay_path)?;
            let beatmap = beatmap::parse_beatmap(&beatmap_path)?;
            
            let result_dict = PyDict::new(py);
            let replay_dict = PyDict::new(py);
            let beatmap_dict = PyDict::new(py);
            
            replay.to_py_dict(replay_dict)?;
            beatmap.to_py_dict(beatmap_dict)?;
            
            result_dict.set_item("replay", replay_dict)?;
            result_dict.set_item("beatmap", beatmap_dict)?;
            
            results.push(result_dict.into());
        }
        
        Ok(results)
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn osu_fast(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_beatmap_fast, m)?)?;
    m.add_function(wrap_pyfunction!(parse_replay_fast, m)?)?;
    m.add_function(wrap_pyfunction!(generate_beatmap_frames_exact, m)?)?;
    m.add_function(wrap_pyfunction!(generate_replay_frames_exact, m)?)?;
    m.add_function(wrap_pyfunction!(batch_parse_files, m)?)?;
    Ok(())
}