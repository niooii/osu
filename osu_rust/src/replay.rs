use crate::types::*;
use pyo3::prelude::*;
use std::fs::File;
use std::io::{Read, Cursor};
use byteorder::{LittleEndian, ReadBytesExt};

pub fn parse_replay(file_path: &str) -> PyResult<ReplayData> {
    let mut file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open replay file: {}", e)))?;
    
    let game_mode = read_byte(&mut file)?;
    
    if game_mode != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a osu!std replay"));
    }
    
    let osu_version = read_int(&mut file)?;
    let map_md5 = read_binary_string(&mut file)?;
    let player = read_binary_string(&mut file)?;
    let replay_md5 = read_binary_string(&mut file)?;
    
    let n_300s = read_short(&mut file)?;
    let n_100s = read_short(&mut file)?;
    let n_50s = read_short(&mut file)?;
    let n_geki = read_short(&mut file)?;
    let n_katu = read_short(&mut file)?;
    let n_misses = read_short(&mut file)?;
    
    let score = read_int(&mut file)?;
    let max_combo = read_short(&mut file)?;
    let perfect = read_byte(&mut file)? == 1;
    
    let total = n_300s + n_100s + n_50s + n_misses;
    let accuracy = if total > 0 {
        (n_300s as f64 + n_100s as f64 / 3.0 + n_50s as f64 / 6.0) / total as f64
    } else {
        0.0
    };
    
    let mods = read_int(&mut file)?;
    
    let life_graph = read_binary_string(&mut file)?;
    let timestamp = read_long(&mut file)?;
    
    let replay_length = read_int(&mut file)?;
    let mut compressed_data = vec![0u8; replay_length as usize];
    file.read_exact(&mut compressed_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read compressed data: {}", e)))?;
    
    let mut decompressed_data = Vec::new();
    lzma_rs::lzma_decompress(&mut Cursor::new(&compressed_data), &mut decompressed_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to decompress replay data: {}", e)))?;
    
    let replay_string = String::from_utf8(decompressed_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to decode replay data: {}", e)))?;
    
    let raw_data: Vec<Vec<&str>> = replay_string
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|t| t.split('|').collect())
        .collect();
    
    let mut parsed_data = Vec::new();
    for parts in raw_data {
        if parts.len() >= 4 {
            if let (Ok(w), Ok(x), Ok(y), Ok(z)) = (
                parts[0].parse::<i32>(),
                parts[1].parse::<f32>(),
                parts[2].parse::<f32>(),
                parts[3].parse::<i32>(),
            ) {
                parsed_data.push((w, x, y, z));
            }
        }
    }
    
    let mut frames = Vec::new();
    let mut offset = 0i32;
    for (w, x, y, z) in parsed_data {
        offset += w;
        frames.push(ReplayFrame {
            time: offset,
            x,
            y,
            keys: z as u32,
        });
    }
    
    frames.sort_by_key(|f| f.time);
    
    let _ = read_long(&mut file)?;
    
    let score_data = ScoreData {
        count_300: n_300s,
        count_100: n_100s,
        count_50: n_50s,
        count_geki: n_geki,
        count_katu: n_katu,
        count_miss: n_misses,
        total_score: score,
        max_combo,
        perfect,
        mods,
        accuracy,
    };
    
    Ok(ReplayData {
        frames,
        game_mode,
        game_version: osu_version,
        beatmap_hash: map_md5,
        player_name: player,
        replay_hash: replay_md5,
        score_data,
        life_graph,
        timestamp,
    })
}

fn read_byte(file: &mut File) -> PyResult<u8> {
    let mut buf = [0u8; 1];
    file.read_exact(&mut buf)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read byte: {}", e)))?;
    Ok(buf[0])
}

fn read_short(file: &mut File) -> PyResult<u16> {
    let b1 = read_byte(file)? as u16;
    let b2 = read_byte(file)? as u16;
    Ok(b1 + (b2 << 8))
}

fn read_int(file: &mut File) -> PyResult<u32> {
    let s1 = read_short(file)? as u32;
    let s2 = read_short(file)? as u32;
    Ok(s1 + (s2 << 16))
}

fn read_long(file: &mut File) -> PyResult<u64> {
    let i1 = read_int(file)? as u64;
    let i2 = read_int(file)? as u64;
    Ok(i1 + (i2 << 32))
}

fn read_uleb128(file: &mut File) -> PyResult<u32> {
    let mut n = 0u32;
    let mut i = 0;
    loop {
        let byte = read_byte(file)?;
        n += ((byte & 0x7F) as u32) << i;
        if byte & 0x80 != 0 {
            i += 7;
        } else {
            return Ok(n);
        }
    }
}

fn read_binary_string(file: &mut File) -> PyResult<String> {
    loop {
        let flag = read_byte(file)?;
        if flag == 0x00 {
            return Ok(String::new());
        } else if flag == 0x0b {
            let length = read_uleb128(file)?;
            let mut buffer = vec![0u8; length as usize];
            file.read_exact(&mut buffer)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read string data: {}", e)))?;
            return String::from_utf8(buffer)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to decode string: {}", e)));
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid string format"));
        }
    }
}

impl ReplayData {
    pub fn frame(&self, time: i32) -> (f32, f32, u32) {
        let index = binary_search(&self.frames, time);
        
        if index >= self.frames.len() {
            return (0.0, 0.0, 0);
        }
        
        let frame = &self.frames[index];
        if frame.time > time {
            if index > 0 {
                let prev_frame = &self.frames[index - 1];
                return (prev_frame.x, prev_frame.y, prev_frame.keys);
            } else {
                return (0.0, 0.0, 0);
            }
        }
        
        (frame.x, frame.y, frame.keys)
    }
}

fn binary_search(frames: &[ReplayFrame], time: i32) -> usize {
    let mut left = 0;
    let mut right = frames.len();
    
    while left < right {
        let mid = left + (right - left) / 2;
        if frames[mid].time <= time {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    if left > 0 && left <= frames.len() {
        left - 1
    } else {
        left
    }
}

pub fn generate_replay_frames_exact(
    replay: &ReplayData,
    start_time: i32,
    end_time: i32,
    frame_rate: i32,
    screen_width: f32,
    screen_height: f32,
) -> Vec<Vec<f32>> {
    let mut frames = Vec::new();
    
    for time in (start_time..end_time).step_by(frame_rate as usize) {
        let (x, y, _) = replay.frame(time);
        let normalized_x = (x / screen_width).max(0.0).min(1.0) - 0.5;
        let normalized_y = (y / screen_height).max(0.0).min(1.0) - 0.5;
        frames.push(vec![normalized_x, normalized_y]);
    }
    
    frames
}