import math
import os

import numpy as np
import pygame
import mutagen.mp3

from osu.rulesets.core import SCREEN_HEIGHT, SCREEN_WIDTH
from osu.preview import beatmap as beatmap_preview
from osu.rulesets import beatmap as bm


def preview_replay(ia_replay, beatmap: bm.Beatmap, audio_file=None):
    """
    Preview a replay with beatmap visualization

    Args:
        ia_replay: numpy array of replay data
        beatmap: loaded beatmap object
        audio_file: path to audio file (optional)
    """

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

    if audio_file and os.path.exists(audio_file):
        pygame.mixer.music.play(start=beatmap['AudioLeadIn'] / 1000)

    running = True
    paused = False
    
    # Progress bar variables
    dragging_progress = False
    was_paused_before_drag = False
    progress_start_time = 0  # Time when first object becomes visible
    progress_end_time = 0    # Time when last object is hit
    
    # Calculate progress bar timing
    if beatmap.hit_objects:
        # Find first visible object time (considering approach time)
        first_obj = beatmap.hit_objects[0]
        preempt, _ = beatmap.approach_rate()
        progress_start_time = first_obj.time - preempt
        
        # Find last object hit time
        last_obj = beatmap.hit_objects[-1]
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
                ox, oy = visible_objects[0].target_position(time, beatmap.beat_duration(time), beatmap['SliderMultiplier'])

                if delta > 0:
                    cx += (ox - cx) / delta
                    cy += (oy - cy) / delta
                else:
                    cx = ox
                    cy = oy

        # cx, cy, z = my_replay.frame(time)
        # pygame.draw.circle(screen, (255, 0, 0), (int(cx), int(cy)), 8)

        frame = int((time - beatmap.start_offset()) // REPLAY_SAMPLING_RATE)
        if frame > 0 and frame < len(ia_replay):
            x, y = ia_replay[frame]
            x += 0.5
            y += 0.5
            x *= SCREEN_WIDTH
            y *= SCREEN_HEIGHT
            pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), 8)

            # Update trail for seeking/dragging
            if dragging_progress:
                # Calculate trail for current position during seeking (same approach as normal trail)
                trail = []
                # Build trail from current frame backwards
                for i in range(8):
                    trail_frame = int(frame - i)
                    if trail_frame > 0 and trail_frame < len(ia_replay):
                        tx, ty = ia_replay[trail_frame]
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