#!/usr/bin/env python3
"""
Test script to verify slider position continuity after the fix
"""

import sys
import os
sys.path.append('.')

import osu.dataset as dataset
import osu.rulesets.beatmap as osu_beatmap
import pandas as pd

def test_slider_continuity(beatmap_path):
    """Test that slider positions change continuously over time"""
    print(f"Testing slider continuity for: {beatmap_path}")
    
    # Load beatmap
    beatmap = osu_beatmap.load(beatmap_path)
    print(f"Loaded beatmap: {beatmap.title()} [{beatmap.version()}]")
    print(f"Number of hit objects: {len(beatmap.effective_hit_objects)}")
    
    # Count sliders
    slider_count = sum(1 for obj in beatmap.effective_hit_objects 
                      if hasattr(obj, 'slider_type'))
    print(f"Number of sliders: {slider_count}")
    
    if slider_count == 0:
        print("No sliders found in this beatmap. Skipping test.")
        return
    
    # Generate input data using the new logic
    df = pd.DataFrame([beatmap], columns=['beatmap'])
    input_data = dataset.input_data(df, verbose=False)
    
    print(f"Generated {len(input_data)} frames")
    
    # Find frames where is_slider = 1
    slider_frames = input_data[input_data['is_slider'] == 1.0]
    print(f"Found {len(slider_frames)} slider frames")
    
    if len(slider_frames) == 0:
        print("No slider frames found. This might indicate an issue.")
        return
    
    # Group consecutive slider frames and check position continuity
    slider_positions = slider_frames[['x', 'y']].values
    
    # Check for position changes
    position_changes = 0
    total_frames = len(slider_positions)
    
    prev_pos = None
    static_count = 0
    change_count = 0
    
    print("\nAnalyzing slider position continuity...")
    print("First 10 slider frame positions:")
    
    for i in range(min(10, len(slider_positions))):
        x, y = slider_positions[i]
        print(f"Frame {i}: x={x:.4f}, y={y:.4f}")
        
        if prev_pos is not None:
            if abs(x - prev_pos[0]) < 0.001 and abs(y - prev_pos[1]) < 0.001:
                static_count += 1
            else:
                change_count += 1
        
        prev_pos = (x, y)
    
    print(f"\nPosition analysis:")
    print(f"Static positions: {static_count}")
    print(f"Changing positions: {change_count}")
    
    if change_count > 0:
        print("✅ PASS: Slider positions are changing over time!")
    else:
        print("❌ FAIL: All slider positions are static!")
    
    return change_count > 0

def find_beatmap_with_sliders():
    """Find a beatmap file with sliders for testing"""
    # Try to find beatmap files in common locations
    test_paths = [
        "test_beatmaps",
        "../test_data", 
        "data/beatmaps",
        "."
    ]
    
    for path in test_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.osu'):
                    full_path = os.path.join(path, file)
                    try:
                        beatmap = osu_beatmap.load(full_path)
                        slider_count = sum(1 for obj in beatmap.effective_hit_objects 
                                         if hasattr(obj, 'slider_type'))
                        if slider_count > 0:
                            return full_path
                    except:
                        continue
    return None

if __name__ == "__main__":
    # Test with a specific beatmap if provided
    if len(sys.argv) > 1:
        beatmap_path = sys.argv[1]
        if not os.path.exists(beatmap_path):
            print(f"Error: Beatmap file not found: {beatmap_path}")
            sys.exit(1)
    else:
        # Try to find a beatmap with sliders
        beatmap_path = find_beatmap_with_sliders()
        if not beatmap_path:
            print("No beatmap files with sliders found.")
            print("Usage: python test_slider_continuity.py <beatmap_file.osu>")
            sys.exit(1)
    
    success = test_slider_continuity(beatmap_path)
    sys.exit(0 if success else 1)