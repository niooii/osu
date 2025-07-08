use crate::types::*;
use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;

pub fn parse_beatmap(file_path: &str) -> PyResult<BeatmapData> {
    let file = File::open(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open beatmap file: {}", e)))?;
    
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // Skip version line
    if let Some(Ok(_version)) = lines.next() {}
    
    let mut sections: HashMap<String, Vec<String>> = HashMap::new();
    let mut current_section = String::new();
    
    // Parse sections
    for line in lines {
        let line = line.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read line: {}", e)))?;
        let line = line.trim();
        
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        
        if line.starts_with('[') && line.ends_with(']') {
            current_section = line[1..line.len()-1].to_string();
            sections.insert(current_section.clone(), Vec::new());
        } else if !current_section.is_empty() {
            sections.get_mut(&current_section).unwrap().push(line.to_string());
        }
    }
    
    // Parse difficulty settings
    let mut approach_rate = 5.0;
    let mut circle_size = 4.0;
    let mut hp_drain_rate = 5.0;
    let mut overall_difficulty = 5.0;
    let mut slider_multiplier = 1.4;
    let mut slider_tick_rate = 1.0;
    let mut audio_lead_in = 0;
    
    if let Some(difficulty_lines) = sections.get("Difficulty") {
        for line in difficulty_lines {
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim();
                let value = value.trim();
                
                match key {
                    "ApproachRate" => approach_rate = value.parse().unwrap_or(5.0),
                    "CircleSize" => circle_size = value.parse().unwrap_or(4.0),
                    "HPDrainRate" => hp_drain_rate = value.parse().unwrap_or(5.0),
                    "OverallDifficulty" => overall_difficulty = value.parse().unwrap_or(5.0),
                    "SliderMultiplier" => slider_multiplier = value.parse().unwrap_or(1.4),
                    "SliderTickRate" => slider_tick_rate = value.parse().unwrap_or(1.0),
                    _ => {}
                }
            }
        }
    }
    
    if let Some(general_lines) = sections.get("General") {
        for line in general_lines {
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim();
                let value = value.trim();
                
                if key == "AudioLeadIn" {
                    audio_lead_in = value.parse().unwrap_or(0);
                }
            }
        }
    }
    
    // Parse timing points
    let mut timing_points = Vec::new();
    if let Some(timing_lines) = sections.get("TimingPoints") {
        for line in timing_lines {
            if let Ok(timing_point) = parse_timing_point(line) {
                timing_points.push(timing_point);
            }
        }
    }
    
    // Parse hit objects with full slider support
    let mut hit_objects = Vec::new();
    if let Some(hitobject_lines) = sections.get("HitObjects") {
        for line in hitobject_lines {
            match parse_hit_object(line) {
                Ok(hit_object) => hit_objects.push(hit_object),
                Err(_) => {
                    // Continue parsing other objects instead of failing completely
                },
            }
        }
    }
    
    Ok(BeatmapData {
        hit_objects,
        timing_points,
        approach_rate,
        circle_size,
        hp_drain_rate,
        overall_difficulty,
        slider_multiplier,
        slider_tick_rate,
        audio_lead_in,
    })
}

fn parse_timing_point(line: &str) -> Result<TimingPoint, Box<dyn std::error::Error>> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 2 {
        return Err("Invalid timing point format".into());
    }
    
    let time = parts[0].parse::<f64>()? as i32;
    let beat_length = parts[1].parse::<f64>()?;
    let meter = if parts.len() > 2 { parts[2].parse().unwrap_or(4) } else { 4 };
    let sample_set = if parts.len() > 3 { parts[3].parse().unwrap_or(0) } else { 0 };
    let sample_index = if parts.len() > 4 { parts[4].parse().unwrap_or(0) } else { 0 };
    let volume = if parts.len() > 5 { parts[5].parse().unwrap_or(100) } else { 100 };
    let uninherited = if parts.len() > 6 { parts[6] == "1" } else { true };
    let effects = if parts.len() > 7 { parts[7].parse().unwrap_or(0) } else { 0 };
    
    Ok(TimingPoint {
        time,
        beat_length,
        meter,
        sample_set,
        sample_index,
        volume,
        uninherited,
        effects,
    })
}

fn parse_hit_object(line: &str) -> Result<HitObject, Box<dyn std::error::Error>> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 4 {
        return Err("Invalid hit object format".into());
    }
    
    let x = parts[0].parse::<f32>()?;
    let y = parts[1].parse::<f32>()?;
    let time = parts[2].parse::<i32>()?;
    let object_type = parts[3].parse::<u8>()?;
    let hit_sound = if parts.len() > 4 { parts[4].parse().unwrap_or(0) } else { 0 };
    
    let mut end_time = None;
    let mut slider_data = None;
    
    // Check if it's a slider (type & 2 != 0)
    if object_type & 2 != 0 && parts.len() > 5 {
        // Parse slider data
        let slider_parts: Vec<&str> = parts[5].split('|').collect();
        if !slider_parts.is_empty() {
            let curve_type_char = slider_parts[0].chars().next().unwrap_or('L');
            let slider_type = SliderType::from_char(curve_type_char);
            
            let mut curve_points = Vec::new();
            
            for i in 1..slider_parts.len() {
                if let Some((px, py)) = slider_parts[i].split_once(':') {
                    if let (Ok(px), Ok(py)) = (px.parse::<f32>(), py.parse::<f32>()) {
                        curve_points.push((px, py));
                    }
                }
            }
            
            let repeat = if parts.len() > 6 { parts[6].parse().unwrap_or(1) } else { 1 };
            let pixel_length = if parts.len() > 7 { parts[7].parse().unwrap_or(100.0) } else { 100.0 };
            
            slider_data = Some(SliderData {
                slider_type,
                curve_points,
                repeat,
                pixel_length,
            });
        }
    }
    
    // Check if it's a spinner (type & 8 != 0)
    if object_type & 8 != 0 && parts.len() > 5 {
        end_time = Some(parts[5].parse().unwrap_or(time + 1000));
    }
    
    Ok(HitObject {
        x,
        y,
        time,
        object_type,
        hit_sound,
        end_time,
        slider_data,
    })
}

pub fn generate_beatmap_frames_exact(
    beatmap: &mut BeatmapData,
    start_time: i32,
    end_time: i32,
    frame_rate: i32,
    screen_width: f32,
    screen_height: f32,
) -> Vec<Vec<f32>> {
    let mut frames = Vec::new();
    let preempt = calculate_preempt(beatmap.approach_rate);
    let mut last_ok_frame: Option<(f32, f32, f32, bool, bool)> = None;
    
    for time in (start_time..end_time).step_by(frame_rate as usize) {
        let frame_result = beatmap_frame(beatmap, time, screen_width, screen_height);
        
        let frame = if let Some((px, py, time_left, is_slider, is_spinner)) = frame_result {
            last_ok_frame = Some((px, py, time_left, is_slider, is_spinner));
            vec![
                px - 0.5,
                py - 0.5,
                if time_left < preempt as f32 { 1.0 } else { 0.0 },
                if is_slider { 1.0 } else { 0.0 },
                if is_spinner { 1.0 } else { 0.0 },
            ]
        } else if let Some((px, py, _, is_slider, is_spinner)) = last_ok_frame {
            vec![
                px - 0.5,
                py - 0.5,
                0.0, // time_left = infinity (not visible)
                if is_slider { 1.0 } else { 0.0 },
                if is_spinner { 1.0 } else { 0.0 },
            ]
        } else {
            // Default frame
            vec![0.0, 0.0, 0.0, 0.0, 0.0]
        };
        
        frames.push(frame);
    }
    
    frames
}

fn beatmap_frame(
    beatmap: &BeatmapData,
    time: i32,
    screen_width: f32,
    screen_height: f32,
) -> Option<(f32, f32, f32, bool, bool)> {
    let visible_objects = beatmap_visible_objects_optimized(beatmap, time, Some(1));
    
    if !visible_objects.is_empty() {
        let obj = &visible_objects[0];
        let beat_duration = beatmap_beat_duration_cached(beatmap, obj.time);
        let (px, py) = obj_target_position_optimized(obj, time, beat_duration, beatmap.slider_multiplier);
        let time_left = (obj.time - time) as f32;
        let is_slider = obj.object_type & 2 != 0;
        let is_spinner = obj.object_type & 8 != 0;
        
        let px = (px / screen_width).max(0.0).min(1.0);
        let py = (py / screen_height).max(0.0).min(1.0);
        
        Some((px, py, time_left, is_slider, is_spinner))
    } else {
        None
    }
}

// OPTIMIZED FUNCTIONS FOR ACCURATE BUT FAST TRAINING DATA

// Optimized object finding with early termination
fn beatmap_visible_objects_optimized(beatmap: &BeatmapData, time: i32, count: Option<usize>) -> Vec<&HitObject> {
    let mut objects = Vec::new();
    let preempt = calculate_preempt(beatmap.approach_rate);
    
    // Early termination - since objects are sorted by time, we can break early
    for obj in &beatmap.hit_objects {
        // If object is too far in the future, break (no more objects will be visible)
        if obj.time - preempt as i32 > time {
            break;
        }
        
        let obj_duration = obj_duration_fast(obj, beatmap, beatmap.slider_multiplier);
        
        // Skip objects that are too far in the past
        if time > obj.time + obj_duration as i32 {
            continue;
        }
        
        // Object is visible
        if time >= obj.time - preempt as i32 && time <= obj.time + obj_duration as i32 {
            objects.push(obj);
            
            if let Some(max_count) = count {
                if objects.len() >= max_count {
                    break;
                }
            }
        }
    }
    
    objects
}

// ACCURATE object duration - no compromises
fn obj_duration_fast(obj: &HitObject, beatmap: &BeatmapData, slider_multiplier: f64) -> f64 {
    if obj.object_type & 2 != 0 {
        // Slider - ACCURATE calculation using proper timing
        if let Some(ref slider_data) = obj.slider_data {
            let beat_duration = beatmap_beat_duration_accurate(beatmap, obj.time);
            beat_duration * slider_data.pixel_length as f64 / (100.0 * slider_multiplier) * slider_data.repeat as f64
        } else {
            0.0
        }
    } else if obj.object_type & 8 != 0 {
        // Spinner - use actual end_time
        if let Some(end_time) = obj.end_time {
            (end_time - obj.time) as f64
        } else {
            0.0 // No default - if no end_time, no duration
        }
    } else {
        // Circle - no duration
        0.0
    }
}

// ACCURATE beat duration calculation - exactly like Python
fn beatmap_beat_duration_cached(beatmap: &BeatmapData, time: i32) -> f64 {
    beatmap_beat_duration_accurate(beatmap, time)
}

fn beatmap_beat_duration_accurate(beatmap: &BeatmapData, time: i32) -> f64 {
    // EXACT implementation from Python _timing() method
    let mut bpm = None;
    let mut timing_point_bpm = 120.0; // Default BPM
    
    // Find the correct timing point (exactly like Python)
    for tp in &beatmap.timing_points {
        if tp.time > time {
            break;
        }
        if tp.uninherited && tp.beat_length > 0.0 {
            bpm = Some(60000.0 / tp.beat_length);
        }
        timing_point_bpm = tp.beat_length;
    }
    
    let final_bpm = bpm.unwrap_or(120.0);
    
    // Apply timing point multiplier exactly like Python
    if timing_point_bpm < 0.0 {
        final_bpm * (-timing_point_bpm / 100.0)
    } else if timing_point_bpm > 0.0 && bpm.is_some() {
        timing_point_bpm
    } else {
        final_bpm
    }
}

// Optimized target position calculation
fn obj_target_position_optimized(obj: &HitObject, time: i32, beat_duration: f64, slider_multiplier: f64) -> (f32, f32) {
    if obj.object_type & 2 != 0 {
        // Slider - use optimized curve calculation
        current_curve_point_optimized(obj, time, beat_duration, slider_multiplier)
    } else {
        // Circle or spinner - return base position
        (obj.x, obj.y)
    }
}

// Optimized curve point calculation
fn current_curve_point_optimized(obj: &HitObject, time: i32, beat_duration: f64, slider_multiplier: f64) -> (f32, f32) {
    let elapsed = time - obj.time;
    if elapsed <= 0 {
        return (obj.x, obj.y);
    }
    
    let duration = obj_duration_accurate(obj, beat_duration, slider_multiplier);
    if elapsed as f64 >= duration {
        // Return the end position
        let curve_points = get_curve_points(obj);
        return if !curve_points.is_empty() {
            curve_points[curve_points.len() - 1]
        } else {
            (obj.x, obj.y)
        };
    }
    
    // Calculate progress along the slider (0.0 to 1.0)
    let progress = elapsed as f64 / duration;
    
    // Get the complete curve points including repeats
    let curve_points = get_curve_points(obj);
    
    if curve_points.is_empty() {
        return (obj.x, obj.y);
    }
    
    if curve_points.len() <= 1 {
        return curve_points[0];
    }
    
    // Arc-length parameterization (optimized with early termination)
    let mut cumulative_lengths = vec![0.0f32];
    let mut total_length = 0.0f32;
    
    for i in 1..curve_points.len() {
        let dx = curve_points[i].0 - curve_points[i-1].0;
        let dy = curve_points[i].1 - curve_points[i-1].1;
        let segment_length = (dx*dx + dy*dy).sqrt();
        total_length += segment_length;
        cumulative_lengths.push(total_length);
    }
    
    let target_length = progress as f32 * total_length;
    
    // Optimized segment finding with early termination
    for i in 0..cumulative_lengths.len() - 1 {
        if cumulative_lengths[i + 1] >= target_length {
            let segment_start_length = cumulative_lengths[i];
            let segment_end_length = cumulative_lengths[i + 1];
            let segment_length = segment_end_length - segment_start_length;
            
            if segment_length > 0.0 {
                let segment_progress = (target_length - segment_start_length) / segment_length;
                let start_point = curve_points[i];
                let end_point = curve_points[i + 1];
                
                let x = start_point.0 + segment_progress * (end_point.0 - start_point.0);
                let y = start_point.1 + segment_progress * (end_point.1 - start_point.1);
                
                return (x, y);
            } else {
                return curve_points[i];
            }
        }
    }
    
    // Fallback: return the last point
    curve_points[curve_points.len() - 1]
}

// Accurate duration calculation for sliders (when we need precision)
fn obj_duration_accurate(obj: &HitObject, beat_duration: f64, slider_multiplier: f64) -> f64 {
    if obj.object_type & 2 != 0 {
        // Slider - calculate duration based on length and velocity
        if let Some(ref slider_data) = obj.slider_data {
            beat_duration * slider_data.pixel_length as f64 / (100.0 * slider_multiplier) * slider_data.repeat as f64
        } else {
            0.0
        }
    } else if obj.object_type & 8 != 0 {
        // Spinner - use end_time if available
        if let Some(end_time) = obj.end_time {
            (end_time - obj.time) as f64
        } else {
            1000.0 // Default spinner duration
        }
    } else {
        // Circle - no duration
        0.0
    }
}

fn beatmap_visible_objects(beatmap: &BeatmapData, time: i32, count: Option<usize>) -> Vec<&HitObject> {
    let mut objects = Vec::new();
    let preempt = calculate_preempt(beatmap.approach_rate);
    
    for obj in &beatmap.hit_objects {
        let obj_duration = obj_duration(obj, beatmap_beat_duration(beatmap, obj.time), beatmap.slider_multiplier);
        
        if time > obj.time + obj_duration as i32 {
            continue;
        } else if time < obj.time - preempt as i32 {
            break;
        } else if time < obj.time + obj_duration as i32 {
            objects.push(obj);
        }
        
        if let Some(max_count) = count {
            if objects.len() >= max_count {
                break;
            }
        }
    }
    
    objects
}

// COMPLEX SLIDER LOGIC WITHOUT CACHING

fn obj_target_position(obj: &HitObject, time: i32, beat_duration: f64, slider_multiplier: f64) -> (f32, f32) {
    if obj.object_type & 2 != 0 {
        // Slider - use proper curve calculation
        current_curve_point(obj, time, beat_duration, slider_multiplier)
    } else {
        // Circle or spinner - return base position
        (obj.x, obj.y)
    }
}

fn current_curve_point(obj: &HitObject, time: i32, beat_duration: f64, slider_multiplier: f64) -> (f32, f32) {
    let elapsed = time - obj.time;
    if elapsed <= 0 {
        return (obj.x, obj.y);
    }
    
    let duration = obj_duration(obj, beat_duration, slider_multiplier);
    if elapsed as f64 >= duration {
        // Return the end position
        let curve_points = get_curve_points(obj);
        return if !curve_points.is_empty() {
            curve_points[curve_points.len() - 1]
        } else {
            (obj.x, obj.y)
        };
    }
    
    // Calculate progress along the slider (0.0 to 1.0)
    let progress = elapsed as f64 / duration;
    
    // Get the complete curve points including repeats
    let curve_points = get_curve_points(obj);
    
    if curve_points.is_empty() {
        return (obj.x, obj.y);
    }
    
    if curve_points.len() <= 1 {
        return curve_points[0];
    }
    
    // Calculate cumulative arc lengths for each point
    let mut cumulative_lengths = vec![0.0f32];
    let mut total_length = 0.0f32;
    
    for i in 1..curve_points.len() {
        let dx = curve_points[i].0 - curve_points[i-1].0;
        let dy = curve_points[i].1 - curve_points[i-1].1;
        let segment_length = (dx*dx + dy*dy).sqrt();
        total_length += segment_length;
        cumulative_lengths.push(total_length);
    }
    
    // Find the target arc length based on progress
    let target_length = progress as f32 * total_length;
    
    // Find the segment containing the target length
    for i in 0..cumulative_lengths.len() - 1 {
        if cumulative_lengths[i] <= target_length && target_length <= cumulative_lengths[i + 1] {
            // Interpolate within this segment
            let segment_start_length = cumulative_lengths[i];
            let segment_end_length = cumulative_lengths[i + 1];
            let segment_length = segment_end_length - segment_start_length;
            
            if segment_length > 0.0 {
                // Calculate interpolation ratio within the segment
                let segment_progress = (target_length - segment_start_length) / segment_length;
                
                // Linear interpolation between the two points
                let start_point = curve_points[i];
                let end_point = curve_points[i + 1];
                
                let x = start_point.0 + segment_progress * (end_point.0 - start_point.0);
                let y = start_point.1 + segment_progress * (end_point.1 - start_point.1);
                
                return (x, y);
            } else {
                // Zero-length segment, return the point
                return curve_points[i];
            }
        }
    }
    
    // Fallback: return the last point
    curve_points[curve_points.len() - 1]
}

fn get_curve_points(obj: &HitObject) -> Vec<(f32, f32)> {
    if let Some(ref slider_data) = obj.slider_data {
        let base_curve = get_base_curve_points(obj);
        
        // Handle repeats and reversals
        let mut complete_curve = Vec::new();
        for i in 1..=slider_data.repeat {
            if i % 2 == 1 {
                // Forward direction
                complete_curve.extend(&base_curve);
            } else {
                // Reverse direction
                let mut reversed = base_curve.clone();
                reversed.reverse();
                complete_curve.extend(reversed);
            }
        }
        
        complete_curve
    } else {
        vec![(obj.x, obj.y)]
    }
}

fn get_base_curve_points(obj: &HitObject) -> Vec<(f32, f32)> {
    if let Some(ref slider_data) = obj.slider_data {
        // Start with the hit object position
        let mut points = vec![(obj.x, obj.y)];
        points.extend(&slider_data.curve_points);
        
        // Generate curve based on type
        let mut curve_points = match slider_data.slider_type {
            SliderType::Linear => compute_linear_curve(&points),
            SliderType::Bezier => compute_bezier_curve(&points),
            SliderType::Perfect => compute_perfect_curve(&points),
            SliderType::Catmull => compute_catmull_curve(&points),
        };
        
        // Adjust curve length to match pixel_length
        let curve_length = calculate_curve_length(&curve_points);
        
        if curve_length < slider_data.pixel_length {
            // Extend the curve to match the target length
            if let Some(last_point) = curve_points.last() {
                let remaining_length = slider_data.pixel_length - curve_length;
                if curve_points.len() >= 2 {
                    // Calculate direction from second-to-last to last point
                    let second_last = curve_points[curve_points.len() - 2];
                    let dx = last_point.0 - second_last.0;
                    let dy = last_point.1 - second_last.1;
                    let length = (dx*dx + dy*dy).sqrt();
                    
                    if length > 0.0 {
                        let unit_x = dx / length;
                        let unit_y = dy / length;
                        
                        // Calculate the end point of the extension
                        let extension_x = last_point.0 + unit_x * remaining_length;
                        let extension_y = last_point.1 + unit_y * remaining_length;
                        
                        // Add the extension point
                        curve_points.push((extension_x, extension_y));
                    }
                }
            }
        } else if curve_length > slider_data.pixel_length {
            // Shorten the curve to match the target length
            let target_length = slider_data.pixel_length;
            let mut current_length = 0.0f32;
            let mut shortened_curve = vec![curve_points[0]]; // Always include the start point
            
            for i in 1..curve_points.len() {
                let dx = curve_points[i].0 - curve_points[i-1].0;
                let dy = curve_points[i].1 - curve_points[i-1].1;
                let segment_length = (dx*dx + dy*dy).sqrt();
                
                if current_length + segment_length <= target_length {
                    // Include the full segment
                    shortened_curve.push(curve_points[i]);
                    current_length += segment_length;
                } else {
                    // Partial segment - interpolate to the target length
                    let remaining_length = target_length - current_length;
                    if segment_length > 0.0 {
                        let ratio = remaining_length / segment_length;
                        let interpolated_x = curve_points[i-1].0 + dx * ratio;
                        let interpolated_y = curve_points[i-1].1 + dy * ratio;
                        shortened_curve.push((interpolated_x, interpolated_y));
                    }
                    break;
                }
            }
            
            curve_points = shortened_curve;
        }
        
        curve_points
    } else {
        vec![(obj.x, obj.y)]
    }
}

fn calculate_curve_length(points: &[(f32, f32)]) -> f32 {
    let mut length = 0.0;
    for i in 1..points.len() {
        let dx = points[i].0 - points[i-1].0;
        let dy = points[i].1 - points[i-1].1;
        length += (dx*dx + dy*dy).sqrt();
    }
    length
}

fn compute_linear_curve(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.len() < 2 {
        return points.to_vec();
    }
    
    let mut curve_points = Vec::new();
    let num_segments = 50; // Fixed number of segments
    
    for i in 0..=num_segments {
        let t = i as f32 / num_segments as f32;
        let x = points[0].0 + t * (points[points.len()-1].0 - points[0].0);
        let y = points[0].1 + t * (points[points.len()-1].1 - points[0].1);
        curve_points.push((x, y));
    }
    
    curve_points
}

fn compute_bezier_curve(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.len() < 2 {
        return points.to_vec();
    }
    
    // EXACT implementation from Python bezier.py
    let num_points = 50;
    let mut curve_points = Vec::new();
    
    // Find segments (osu! Bézier curves are separated by repeated points)
    let mut segments = Vec::new();
    let mut current_segment = vec![points[0]];
    
    for i in 1..points.len() {
        if points[i] == points[i-1] && current_segment.len() > 1 {
            // End of segment, start new one
            segments.push(current_segment);
            current_segment = vec![points[i]];
        } else {
            current_segment.push(points[i]);
        }
    }
    
    // Add the last segment
    if !current_segment.is_empty() {
        segments.push(current_segment);
    }
    
    // Process each segment exactly like Python
    for segment in segments {
        if segment.len() == 1 {
            curve_points.push(segment[0]);
        } else if segment.len() == 2 {
            // Linear interpolation for 2 points
            for t in 0..num_points {
                let t_val = t as f32 / (num_points - 1) as f32;
                let x = segment[0].0 + t_val * (segment[1].0 - segment[0].0);
                let y = segment[0].1 + t_val * (segment[1].1 - segment[0].1);
                curve_points.push((x, y));
            }
        } else if segment.len() == 3 {
            // Quadratic Bézier curve
            for t in 0..num_points {
                let t_val = t as f32 / (num_points - 1) as f32;
                let one_minus_t = 1.0 - t_val;
                let x = one_minus_t * one_minus_t * segment[0].0 + 
                        2.0 * one_minus_t * t_val * segment[1].0 + 
                        t_val * t_val * segment[2].0;
                let y = one_minus_t * one_minus_t * segment[0].1 + 
                        2.0 * one_minus_t * t_val * segment[1].1 + 
                        t_val * t_val * segment[2].1;
                curve_points.push((x, y));
            }
        } else {
            // Cubic or higher degree Bézier - use de Casteljau's algorithm
            for t in 0..num_points {
                let t_val = t as f32 / (num_points - 1) as f32;
                let point = de_casteljau(&segment, t_val);
                curve_points.push(point);
            }
        }
    }
    
    curve_points
}

// De Casteljau's algorithm for Bézier curves of any degree
fn de_casteljau(points: &[(f32, f32)], t: f32) -> (f32, f32) {
    if points.len() == 1 {
        return points[0];
    }
    
    let mut temp_points = points.to_vec();
    
    for level in 0..points.len()-1 {
        for i in 0..temp_points.len()-1 {
            temp_points[i] = (
                (1.0 - t) * temp_points[i].0 + t * temp_points[i + 1].0,
                (1.0 - t) * temp_points[i].1 + t * temp_points[i + 1].1,
            );
        }
        temp_points.pop();
    }
    
    temp_points[0]
}

fn compute_perfect_curve(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.len() < 3 {
        return compute_linear_curve(points);
    }
    
    // Perfect circle calculation - find center and radius from 3 points
    let p1 = points[0];
    let p2 = points[1]; 
    let p3 = points[2];
    
    // Calculate circle center using perpendicular bisectors
    let d = 2.0 * (p1.0 * (p2.1 - p3.1) + p2.0 * (p3.1 - p1.1) + p3.0 * (p1.1 - p2.1));
    
    if d.abs() < 1e-6 {
        // Points are collinear, use linear interpolation
        return compute_linear_curve(points);
    }
    
    let ux = ((p1.0 * p1.0 + p1.1 * p1.1) * (p2.1 - p3.1) + 
              (p2.0 * p2.0 + p2.1 * p2.1) * (p3.1 - p1.1) + 
              (p3.0 * p3.0 + p3.1 * p3.1) * (p1.1 - p2.1)) / d;
    
    let uy = ((p1.0 * p1.0 + p1.1 * p1.1) * (p3.0 - p2.0) + 
              (p2.0 * p2.0 + p2.1 * p2.1) * (p1.0 - p3.0) + 
              (p3.0 * p3.0 + p3.1 * p3.1) * (p2.0 - p1.0)) / d;
    
    let center = (ux, uy);
    let radius = ((p1.0 - center.0) * (p1.0 - center.0) + (p1.1 - center.1) * (p1.1 - center.1)).sqrt();
    
    // Calculate angles for arc
    let start_angle = (p1.1 - center.1).atan2(p1.0 - center.0);
    let end_angle = (p3.1 - center.1).atan2(p3.0 - center.0);
    
    // Determine arc direction
    let mut angle_diff = end_angle - start_angle;
    if angle_diff > std::f32::consts::PI {
        angle_diff -= 2.0 * std::f32::consts::PI;
    } else if angle_diff < -std::f32::consts::PI {
        angle_diff += 2.0 * std::f32::consts::PI;
    }
    
    // Generate arc points
    let num_points = 50;
    let mut curve_points = Vec::new();
    
    for i in 0..=num_points {
        let t = i as f32 / num_points as f32;
        let angle = start_angle + t * angle_diff;
        let x = center.0 + radius * angle.cos();
        let y = center.1 + radius * angle.sin();
        curve_points.push((x, y));
    }
    
    curve_points
}

fn compute_catmull_curve(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.len() < 3 {
        return compute_linear_curve(points);
    }
    
    // Catmull-Rom spline implementation
    let mut curve_points = Vec::new();
    let num_segments = points.len() - 1;
    let points_per_segment = 50 / num_segments.max(1);
    
    for i in 0..num_segments {
        // Get control points for this segment
        let p0 = if i == 0 { points[0] } else { points[i - 1] };
        let p1 = points[i];
        let p2 = points[i + 1];
        let p3 = if i + 2 < points.len() { points[i + 2] } else { points[i + 1] };
        
        // Generate points for this segment
        for j in 0..points_per_segment {
            let t = j as f32 / points_per_segment as f32;
            let t2 = t * t;
            let t3 = t2 * t;
            
            // Catmull-Rom basis functions
            let c0 = -0.5 * t3 + t2 - 0.5 * t;
            let c1 = 1.5 * t3 - 2.5 * t2 + 1.0;
            let c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
            let c3 = 0.5 * t3 - 0.5 * t2;
            
            let x = c0 * p0.0 + c1 * p1.0 + c2 * p2.0 + c3 * p3.0;
            let y = c0 * p0.1 + c1 * p1.1 + c2 * p2.1 + c3 * p3.1;
            
            curve_points.push((x, y));
        }
    }
    
    // Add the final point
    if !curve_points.is_empty() {
        curve_points.push(points[points.len() - 1]);
    }
    
    curve_points
}

fn obj_duration(obj: &HitObject, beat_duration: f64, slider_multiplier: f64) -> f64 {
    if obj.object_type & 2 != 0 {
        // Slider - calculate duration based on length and velocity
        if let Some(ref slider_data) = obj.slider_data {
            beat_duration * slider_data.pixel_length as f64 / (100.0 * slider_multiplier) * slider_data.repeat as f64
        } else {
            0.0
        }
    } else if obj.object_type & 8 != 0 {
        // Spinner - use end_time if available
        if let Some(end_time) = obj.end_time {
            (end_time - obj.time) as f64
        } else {
            1000.0 // Default spinner duration
        }
    } else {
        // Circle - no duration
        0.0
    }
}

fn calculate_preempt(approach_rate: f32) -> f32 {
    if approach_rate <= 5.0 {
        1200.0 + 600.0 * (5.0 - approach_rate) / 5.0
    } else {
        1200.0 - 750.0 * (approach_rate - 5.0) / 5.0
    }
}

fn beatmap_beat_duration(beatmap: &BeatmapData, time: i32) -> f64 {
    // Find the appropriate timing point
    let mut bpm = 120.0;
    let mut current_multiplier = 1.0;
    
    for tp in &beatmap.timing_points {
        if tp.time > time {
            break;
        }
        
        if tp.uninherited {
            // This is a BPM change
            bpm = 60000.0 / tp.beat_length;
        } else {
            // This is a slider velocity change
            current_multiplier = -100.0 / tp.beat_length;
        }
    }
    
    60000.0 / bpm * current_multiplier
}