import math
import os

import numpy as np
import pygame
import mutagen.mp3

from osu.rulesets.core import SCREEN_HEIGHT, SCREEN_WIDTH
from osu.preview import beatmap as beatmap_preview
from osu.rulesets import beatmap as bm
from osu.rulesets import replay as replay_module
from osu import dataset
from osu.rulesets.mods import Mods


def preview_replay_raw(ia_replay, beatmap_path: str, mods=None, audio_file=None):
    """
    Preview a replay with beatmap visualization using raw replay data

    Args:
        ia_replay: numpy array of replay data
        beatmap: loaded beatmap object (will be copied to avoid modification)
        mods: bitflag of mods to apply (e.g., Mods.DOUBLE_TIME | Mods.HARD_ROCK) (optional)
        audio_file: path to audio file (optional)
    """
    
    beatmap = bm.load(beatmap_path)

    # Apply mods to beatmap copy if provided
    if mods:
        beatmap.apply_mods(mods)
        
    # TODO: DT/HT timing effects need proper implementation
    # Should affect game clock speed, audio speed, and replay timing

    # Setup audio
    if audio_file and os.path.exists(audio_file):
        mp3 = mutagen.mp3.MP3(audio_file)
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.set_volume(0.1)

    pygame.init()

    pygame.display.set_caption(f'{beatmap.title()} [{beatmap.version()}]')

    time = 0
    clock = pygame.time.Clock()
    
    # Add progress bar height
    PROGRESS_BAR_HEIGHT = 50
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + PROGRESS_BAR_HEIGHT))

    FRAME_RATE = 1000
    REPLAY_SAMPLING_RATE = 24

    cx = 0
    cy = 0

    preview = beatmap_preview.from_beatmap(beatmap)

    trail = []
    
    # Track key states for press/release detection
    prev_k1 = False
    prev_k2 = False

    if audio_file and os.path.exists(audio_file):
        # pygame doesn't support speed changes, so audio won't match DT/HT timing
        # so kill it for now
        if beatmap.mods & Mods.DOUBLE_TIME or beatmap.mods & Mods.HALF_TIME:
            pass
        else:
            pygame.mixer.music.play(start=beatmap['AudioLeadIn'] / 1000)

    running = True
    paused = False
    
    # Progress bar variables
    dragging_progress = False
    was_paused_before_drag = False
    progress_start_time = 0  # Time when first object becomes visible
    progress_end_time = 0    # Time when last object is hit
    
    # Calculate progress bar timing
    if beatmap.effective_hit_objects:
        # Find first visible object time (considering approach time)
        first_obj = beatmap.effective_hit_objects[0]

        base_preempt, _ = beatmap.base_approach_rate_timing()
        progress_start_time = first_obj.time - base_preempt
        
        # Find last object hit time
        last_obj = beatmap.effective_hit_objects[-1]
        if hasattr(last_obj, 'end_time'):
            progress_end_time = last_obj.end_time
        else:
            progress_end_time = last_obj.time
    
    progress_bar_y = SCREEN_HEIGHT + 10
    progress_bar_height = 4
    progress_bar_margin = 20

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    if audio_file and os.path.exists(audio_file):
                        if paused:
                            pygame.mixer.music.pause()
                        else:
                            if audio_file and os.path.exists(audio_file):
                                # Restart audio from the new position
                                pygame.mixer.music.stop()
                                pygame.mixer.music.play(start=max(0, (time - beatmap['AudioLeadIn']) / 1000))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    # Check if click is on progress bar
                    if (progress_bar_y <= mouse_y <= progress_bar_y + progress_bar_height + 10 and
                        progress_bar_margin <= mouse_x <= SCREEN_WIDTH - progress_bar_margin):
                        dragging_progress = True
                        was_paused_before_drag = paused
                        paused = True
                        if audio_file and os.path.exists(audio_file):
                            pygame.mixer.music.pause()
                        
                        # Calculate new time based on click position
                        progress_ratio = (mouse_x - progress_bar_margin) / (SCREEN_WIDTH - 2 * progress_bar_margin)
                        progress_ratio = max(0, min(1, progress_ratio))
                        time = progress_start_time + progress_ratio * (progress_end_time - progress_start_time)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging_progress:
                    dragging_progress = False
                    if not was_paused_before_drag:
                        paused = False
                        if audio_file and os.path.exists(audio_file):
                            # Restart audio from the new position
                            pygame.mixer.music.stop()
                            pygame.mixer.music.play(start=max(0, (time - beatmap['AudioLeadIn']) / 1000))
            elif event.type == pygame.MOUSEMOTION:
                if dragging_progress:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    # Update time based on drag position
                    progress_ratio = (mouse_x - progress_bar_margin) / (SCREEN_WIDTH - 2 * progress_bar_margin)
                    progress_ratio = max(0, min(1, progress_ratio))
                    time = progress_start_time + progress_ratio * (progress_end_time - progress_start_time)

        if not paused:
            time += clock.get_time()
        
        screen.fill((0, 0, 0))

        preview.render(screen, time)
        
        if not paused:
            visible_objects = beatmap.visible_objects(time)

            if len(visible_objects) > 0:
                delta = visible_objects[0].time - time
                ox, oy = visible_objects[0].target_position(time, beatmap.beat_duration(time), beatmap.slider_multiplier())

                if delta > 0:
                    cx += (ox - cx) / delta
                    cy += (oy - cy) / delta
                else:
                    cx = ox
                    cy = oy

        # cx, cy, z = my_replay.frame(time)
        # pygame.draw.circle(screen, (255, 0, 0), (int(cx), int(cy)), 8)

        frame = int((time - beatmap.start_offset()) // REPLAY_SAMPLING_RATE)
        # print(f'we on frame {frame}')
        if 0 < frame < len(ia_replay):
            x, y, k1, k2 = ia_replay[frame]
            x += 0.5
            y += 0.5
            x *= SCREEN_WIDTH
            y *= SCREEN_HEIGHT
            pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), 8)
            
            # Detect key press/release events
            if k1 and not prev_k1:
                print("k1 pressed")
            elif not k1 and prev_k1:
                print("k1 released")
            
            if k2 and not prev_k2:
                print("k2 pressed")
            elif not k2 and prev_k2:
                print("k2 released")
            
            # Update previous key states
            prev_k1 = k1
            prev_k2 = k2

            # Update trail for seeking/dragging
            if dragging_progress:
                # Calculate trail for current position during seeking (same approach as normal trail)
                trail = []
                # Build trail from current frame backwards
                for i in range(8):
                    trail_frame = int(frame - i)
                    if trail_frame > 0 and trail_frame < len(ia_replay):
                        tx, ty, _, _ = ia_replay[trail_frame]
                        tx += 0.5
                        ty += 0.5
                        tx *= SCREEN_WIDTH
                        ty *= SCREEN_HEIGHT
                        trail.insert(0, (tx, ty))  # Insert at beginning to maintain order
                    else:
                        break

            trail_surface = pygame.Surface((screen.get_width(), screen.get_height()))
            trail_surface.set_colorkey((0, 0, 0))
            for tx, ty in trail:
                pygame.draw.circle(trail_surface, (0, 255, 0), (int(tx), int(ty)), 6)
            trail_surface.set_alpha(127)
            screen.blit(trail_surface, (0, 0))

            if not paused and not dragging_progress:
                trail.append((x, y))
                if len(trail) > 64:
                    trail.pop(0)

        # Draw progress bar
        if progress_end_time > progress_start_time:
            # Draw grey progress bar background
            pygame.draw.rect(screen, (128, 128, 128), 
                           (progress_bar_margin, progress_bar_y, 
                            SCREEN_WIDTH - 2 * progress_bar_margin, progress_bar_height))
            
            # Calculate progress position
            progress_ratio = (time - progress_start_time) / (progress_end_time - progress_start_time)
            progress_ratio = max(0, min(1, progress_ratio))
            
            # Draw white progress marker
            marker_x = progress_bar_margin + progress_ratio * (SCREEN_WIDTH - 2 * progress_bar_margin)
            pygame.draw.rect(screen, (255, 255, 255), 
                           (marker_x - 2, progress_bar_y - 2, 4, progress_bar_height + 4))

        pygame.display.flip()

        clock.tick(FRAME_RATE)

    pygame.quit()


def preview_replay(replay: replay_module.Replay, beatmap_path: str, audio_file=None):
    """
    Preview a replay with beatmap visualization using a Replay object
    
    Args:
        replay: Replay object loaded from .osr file
        beatmap_path: Path to the beatmap file
        audio_file: path to audio file (optional)
    """

    beatmap = bm.load(beatmap_path)
    
    # Extract mods from replay
    mods = 0
    if replay.has_mods(replay_module.Mod.DT):
        print("HAS DT")
        mods |= Mods.DOUBLE_TIME
    if replay.has_mods(replay_module.Mod.HR):
        mods |= Mods.HARD_ROCK
    if replay.has_mods(replay_module.Mod.EZ):
        mods |= Mods.EASY
    if replay.has_mods(replay_module.Mod.HT):
        mods |= Mods.HALF_TIME

    beatmap.apply_mods(mods)

    replay_data = dataset.replay_to_output_data(beatmap, replay)

    # Flatten
    ia_replay = []
    for chunk in replay_data:
        for frame in chunk:
            x, y, k1, k2 = frame[0], frame[1], frame[2], frame[3]
            ia_replay.append([x, y, k1, k2])
    
    ia_replay = np.array(ia_replay)
    
    preview_replay_raw(ia_replay, beatmap_path=beatmap_path, mods=mods, audio_file=audio_file)


def preview_training_data(xs, ys, audio_file=None):
    """
    Preview training data with 3 green circles from xs data and cursor from ys data
    
    Args:
        xs: numpy array of beatmap/input data, shape (batches, frames, 9)
        ys: numpy array of replay/output data, shape (batches, frames, 4)  
        audio_file: path to audio file (optional)
    """
    
    # Setup audio
    if audio_file and os.path.exists(audio_file):
        mp3 = mutagen.mp3.MP3(audio_file)
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.set_volume(0.1)

    pygame.init()

    pygame.display.set_caption('Training Data Preview')

    time = 0
    clock = pygame.time.Clock()
    
    # Add progress bar height
    PROGRESS_BAR_HEIGHT = 50
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + PROGRESS_BAR_HEIGHT))

    FRAME_RATE = 1000
    REPLAY_SAMPLING_RATE = 24

    # Flatten data for easier access
    xs_flat = xs.reshape(-1, xs.shape[-1])  # (total_frames, 9)
    ys_flat = ys.reshape(-1, ys.shape[-1])  # (total_frames, 4)
    
    total_frames = len(xs_flat)
    
    trail = []
    
    # Track key states for press/release detection
    prev_k1 = False
    prev_k2 = False

    if audio_file and os.path.exists(audio_file):
        pygame.mixer.music.play()

    running = True
    paused = False
    
    # Progress bar variables
    dragging_progress = False
    was_paused_before_drag = False
    progress_start_time = 0
    progress_end_time = total_frames * REPLAY_SAMPLING_RATE  # Total duration in ms
    
    progress_bar_y = SCREEN_HEIGHT + 10
    progress_bar_height = 4
    progress_bar_margin = 20

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    if audio_file and os.path.exists(audio_file):
                        if paused:
                            pygame.mixer.music.pause()
                        else:
                            pygame.mixer.music.unpause()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    # Check if click is on progress bar
                    if (progress_bar_y <= mouse_y <= progress_bar_y + progress_bar_height + 10 and
                        progress_bar_margin <= mouse_x <= SCREEN_WIDTH - progress_bar_margin):
                        dragging_progress = True
                        was_paused_before_drag = paused
                        paused = True
                        if audio_file and os.path.exists(audio_file):
                            pygame.mixer.music.pause()
                        
                        # Calculate new time based on click position
                        progress_ratio = (mouse_x - progress_bar_margin) / (SCREEN_WIDTH - 2 * progress_bar_margin)
                        progress_ratio = max(0, min(1, progress_ratio))
                        time = progress_ratio * progress_end_time
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging_progress:
                    dragging_progress = False
                    if not was_paused_before_drag:
                        paused = False
                        if audio_file and os.path.exists(audio_file):
                            pygame.mixer.music.unpause()
            elif event.type == pygame.MOUSEMOTION:
                if dragging_progress:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    # Update time based on drag position
                    progress_ratio = (mouse_x - progress_bar_margin) / (SCREEN_WIDTH - 2 * progress_bar_margin)
                    progress_ratio = max(0, min(1, progress_ratio))
                    time = progress_ratio * progress_end_time

        if not paused:
            time += clock.get_time()
        
        screen.fill((0, 0, 0))

        # Calculate current frame
        frame = int(time // REPLAY_SAMPLING_RATE)
        
        if 0 <= frame < total_frames:
            # Draw 3 green circles from xs data (current, previous, next frames)
            for offset in [-1, 0, 1]:
                target_frame = frame + offset
                if 0 <= target_frame < total_frames:
                    # Check if any object type flag is active (is_slider, is_spinner, is_note)
                    is_slider = xs_flat[target_frame, 3]
                    is_spinner = xs_flat[target_frame, 4]
                    is_note = xs_flat[target_frame, 5]
                    
                    # Only draw if at least one object type is active
                    if is_slider or is_spinner or is_note:
                        # Extract x, y from xs data (first 2 columns)
                        xs_x, xs_y = xs_flat[target_frame, 0], xs_flat[target_frame, 1]
                        
                        # Convert from normalized coordinates (-0.5 to 0.5) to screen coordinates
                        xs_x += 0.5
                        xs_y += 0.5
                        xs_x *= SCREEN_WIDTH
                        xs_y *= SCREEN_HEIGHT
                        
                        # Different circle sizes for visual distinction
                        if offset == 0:  # Current frame
                            pygame.draw.circle(screen, (0, 255, 0), (int(xs_x), int(xs_y)), 12)
                        else:  # Previous/next frames
                            pygame.draw.circle(screen, (0, 200, 0), (int(xs_x), int(xs_y)), 8)

            # Draw cursor from ys data (same logic as original)
            if 0 <= frame < len(ys_flat):
                x, y, k1, k2 = ys_flat[frame]
                x += 0.5
                y += 0.5
                x *= SCREEN_WIDTH
                y *= SCREEN_HEIGHT
                pygame.draw.circle(screen, (255, 255, 0), (int(x), int(y)), 8)  # Yellow cursor
                
                # Detect key press/release events
                if k1 and not prev_k1:
                    print("k1 pressed")
                elif not k1 and prev_k1:
                    print("k1 released")
                
                if k2 and not prev_k2:
                    print("k2 pressed")
                elif not k2 and prev_k2:
                    print("k2 released")
                
                # Update previous key states
                prev_k1 = k1
                prev_k2 = k2

                # Update trail for seeking/dragging
                if dragging_progress:
                    # Calculate trail for current position during seeking
                    trail = []
                    # Build trail from current frame backwards
                    for i in range(8):
                        trail_frame = int(frame - i)
                        if trail_frame >= 0 and trail_frame < len(ys_flat):
                            tx, ty, _, _ = ys_flat[trail_frame]
                            tx += 0.5
                            ty += 0.5
                            tx *= SCREEN_WIDTH
                            ty *= SCREEN_HEIGHT
                            trail.insert(0, (tx, ty))  # Insert at beginning to maintain order
                        else:
                            break

                trail_surface = pygame.Surface((screen.get_width(), screen.get_height()))
                trail_surface.set_colorkey((0, 0, 0))
                for tx, ty in trail:
                    pygame.draw.circle(trail_surface, (255, 255, 0), (int(tx), int(ty)), 6)
                trail_surface.set_alpha(127)
                screen.blit(trail_surface, (0, 0))

                if not paused and not dragging_progress:
                    trail.append((x, y))
                    if len(trail) > 64:
                        trail.pop(0)

        # Draw progress bar
        if progress_end_time > progress_start_time:
            # Draw grey progress bar background
            pygame.draw.rect(screen, (128, 128, 128), 
                           (progress_bar_margin, progress_bar_y, 
                            SCREEN_WIDTH - 2 * progress_bar_margin, progress_bar_height))
            
            # Calculate progress position
            progress_ratio = time / progress_end_time
            progress_ratio = max(0, min(1, progress_ratio))
            
            # Draw white progress marker
            marker_x = progress_bar_margin + progress_ratio * (SCREEN_WIDTH - 2 * progress_bar_margin)
            pygame.draw.rect(screen, (255, 255, 255), 
                           (marker_x - 2, progress_bar_y - 2, 4, progress_bar_height + 4))

        pygame.display.flip()

        clock.tick(FRAME_RATE)

    pygame.quit()