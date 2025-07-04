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
        pygame.mixer.init(frequency=mp3.info.sample_rate)
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.set_volume(0.1)

    pygame.init()

    time = 0
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    FRAME_RATE = 1000
    REPLAY_SAMPLING_RATE = 24

    cx = 0
    cy = 0

    preview = beatmap_preview.from_beatmap(beatmap)

    trail = []

    if audio_file and os.path.exists(audio_file):
        pygame.mixer.music.play(start=beatmap['AudioLeadIn'] / 1000)

    running = True

    while running:
        if audio_file and os.path.exists(audio_file):
            time = pygame.mixer.music.get_pos()
        else:
            time += clock.get_time()

        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        preview.render(screen, time)

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

        frame = (time - beatmap.start_offset()) // REPLAY_SAMPLING_RATE
        if frame > 0 and frame < len(ia_replay):
            x, y = ia_replay[frame]
            x += 0.5
            y += 0.5
            x *= SCREEN_WIDTH
            y *= SCREEN_HEIGHT
            pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), 8)

            trail_surface = pygame.Surface((screen.get_width(), screen.get_height()))
            trail_surface.set_colorkey((0, 0, 0))
            for tx, ty in trail:
                pygame.draw.circle(trail_surface, (0, 255, 0), (int(tx), int(ty)), 6)
            trail_surface.set_alpha(127)
            screen.blit(trail_surface, (0, 0))
            trail.append((x, y))
            if len(trail) > 64:
                trail.pop(0)

        pygame.display.flip()

        clock.tick(FRAME_RATE)

    pygame.quit()
