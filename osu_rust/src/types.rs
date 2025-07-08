use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitObject {
    pub x: f32,
    pub y: f32,
    pub time: i32,
    pub object_type: u8,
    pub hit_sound: u8,
    pub end_time: Option<i32>,
    pub slider_data: Option<SliderData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SliderType {
    Linear,   // L
    Bezier,   // B
    Perfect,  // P
    Catmull,  // C
}

impl SliderType {
    pub fn from_char(c: char) -> Self {
        match c {
            'L' => SliderType::Linear,
            'B' => SliderType::Bezier,
            'P' => SliderType::Perfect,
            'C' => SliderType::Catmull,
            _ => SliderType::Linear, // fallback
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliderData {
    pub slider_type: SliderType,
    pub curve_points: Vec<(f32, f32)>,
    pub repeat: i32,
    pub pixel_length: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingPoint {
    pub time: i32,
    pub beat_length: f64,
    pub meter: i32,
    pub sample_set: i32,
    pub sample_index: i32,
    pub volume: i32,
    pub uninherited: bool,
    pub effects: i32,
}

#[derive(Debug, Clone)]
pub struct BeatmapData {
    pub hit_objects: Vec<HitObject>,
    pub timing_points: Vec<TimingPoint>,
    pub approach_rate: f32,
    pub circle_size: f32,
    pub hp_drain_rate: f32,
    pub overall_difficulty: f32,
    pub slider_multiplier: f64,
    pub slider_tick_rate: f64,
    pub audio_lead_in: i32,
}

#[derive(Debug, Clone)]
pub struct ReplayFrame {
    pub time: i32,
    pub x: f32,
    pub y: f32,
    pub keys: u32,
}

#[derive(Debug, Clone)]
pub struct ReplayData {
    pub frames: Vec<ReplayFrame>,
    pub game_mode: u8,
    pub game_version: u32,
    pub beatmap_hash: String,
    pub player_name: String,
    pub replay_hash: String,
    pub score_data: ScoreData,
    pub life_graph: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct ScoreData {
    pub count_300: u16,
    pub count_100: u16,
    pub count_50: u16,
    pub count_geki: u16,
    pub count_katu: u16,
    pub count_miss: u16,
    pub total_score: u32,
    pub max_combo: u16,
    pub perfect: bool,
    pub mods: u32,
    pub accuracy: f64,
}

impl BeatmapData {
    pub fn from_py_dict(_dict: &PyDict) -> PyResult<Self> {
        // Implementation to convert PyDict to BeatmapData
        // This would extract the relevant fields from the Python beatmap object
        todo!("Implement conversion from Python dict")
    }

    pub fn to_py_dict(&self, dict: &PyDict) -> PyResult<()> {
        use pyo3::types::PyList;
        
        // Convert hit objects to Python list - EXACT FORMAT MATCH
        Python::with_gil(|py| {
            let py_hit_objects = PyList::empty(py);
            for obj in &self.hit_objects {
                let obj_list = PyList::empty(py);
                
                // Format exactly like Python expects: [x, y, time, type, hit_sound, ...]
                obj_list.append(obj.x as i32)?;  // Python expects int
                obj_list.append(obj.y as i32)?;  // Python expects int  
                obj_list.append(obj.time)?;
                obj_list.append(obj.object_type)?;
                obj_list.append(obj.hit_sound)?;
                
                // For sliders, append slider data in Python format
                if obj.object_type & 2 != 0 {  // Is slider
                    if let Some(slider) = &obj.slider_data {
                        let mut curve_str = match slider.slider_type {
                            SliderType::Linear => "L".to_string(),
                            SliderType::Bezier => "B".to_string(),
                            SliderType::Perfect => "P".to_string(),
                            SliderType::Catmull => "C".to_string(),
                        };
                        
                        for (x, y) in &slider.curve_points {
                            curve_str.push_str(&format!("|{}:{}", x, y));
                        }
                        
                        obj_list.append(curve_str)?;                    // index 5
                        obj_list.append(slider.repeat)?;                // index 6  
                        obj_list.append(slider.pixel_length)?;          // index 7
                    } else {
                        // Slider without data - add minimal data
                        obj_list.append("L")?;                          // Linear type
                        obj_list.append(1)?;                            // 1 repeat
                        obj_list.append(100.0)?;                        // 100px length
                    }
                }
                
                // For spinners, append end_time
                if obj.object_type & 8 != 0 {  // Is spinner
                    if let Some(end_time) = obj.end_time {
                        obj_list.append(end_time)?;                     // index 5
                    } else {
                        obj_list.append(obj.time + 1000)?;              // Default 1s duration
                    }
                }
                
                py_hit_objects.append(obj_list)?;
            }
            dict.set_item("hit_objects", py_hit_objects)?;
            
            // Convert timing points
            let py_timing_points = PyList::empty(py);
            for tp in &self.timing_points {
                let tp_list = PyList::empty(py);
                tp_list.append(tp.time)?;
                tp_list.append(tp.beat_length)?;
                tp_list.append(tp.meter)?;
                tp_list.append(tp.sample_set)?;
                tp_list.append(tp.sample_index)?;
                tp_list.append(tp.volume)?;
                tp_list.append(tp.uninherited)?;
                tp_list.append(tp.effects)?;
                py_timing_points.append(tp_list)?;
            }
            dict.set_item("timing_points", py_timing_points)?;
            
            // Set difficulty values
            dict.set_item("ApproachRate", self.approach_rate)?;
            dict.set_item("CircleSize", self.circle_size)?;
            dict.set_item("HPDrainRate", self.hp_drain_rate)?;
            dict.set_item("OverallDifficulty", self.overall_difficulty)?;
            dict.set_item("SliderMultiplier", self.slider_multiplier)?;
            dict.set_item("SliderTickRate", self.slider_tick_rate)?;
            dict.set_item("AudioLeadIn", self.audio_lead_in)?;
            
            Ok(())
        })
    }
}

impl ReplayData {
    pub fn from_py_dict(_dict: &PyDict) -> PyResult<Self> {
        // Implementation to convert PyDict to ReplayData
        todo!("Implement conversion from Python dict")
    }

    pub fn to_py_dict(&self, dict: &PyDict) -> PyResult<()> {
        use pyo3::types::PyList;
        
        Python::with_gil(|py| {
            // Convert frames to Python format: [(offset, x, y, z), ...]
            let py_frames = PyList::empty(py);
            for frame in &self.frames {
                let frame_tuple = (frame.time, frame.x, frame.y, frame.keys);
                py_frames.append(frame_tuple)?;
            }
            dict.set_item("data", py_frames)?;
            
            // Set replay metadata
            dict.set_item("game_mode", self.game_mode)?;
            dict.set_item("osu_version", self.game_version)?;
            dict.set_item("map_md5", &self.beatmap_hash)?;
            dict.set_item("player", &self.player_name)?;
            dict.set_item("replay_md5", &self.replay_hash)?;
            dict.set_item("timestamp", self.timestamp)?;
            dict.set_item("life_graph", &self.life_graph)?;
            
            // Set score data
            dict.set_item("n_300s", self.score_data.count_300)?;
            dict.set_item("n_100s", self.score_data.count_100)?;
            dict.set_item("n_50s", self.score_data.count_50)?;
            dict.set_item("n_geki", self.score_data.count_geki)?;
            dict.set_item("n_katu", self.score_data.count_katu)?;
            dict.set_item("n_misses", self.score_data.count_miss)?;
            dict.set_item("score", self.score_data.total_score)?;
            dict.set_item("max_combo", self.score_data.max_combo)?;
            dict.set_item("perfect", self.score_data.perfect)?;
            dict.set_item("mods_int", self.score_data.mods)?;
            
            Ok(())
        })
    }
}