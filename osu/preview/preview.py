import math
import os
import subprocess

import numpy as np
import pygame
import mutagen.mp3
import tqdm.auto as tqdm

from osu.rulesets.core import SCREEN_HEIGHT, SCREEN_WIDTH, REPLAY_SAMPLING_RATE
from osu.preview import beatmap as beatmap_preview
from osu.rulesets import beatmap as bm
from osu.rulesets import replay as replay_module
from osu import dataset
from osu.rulesets.mods import Mods


def _compute_cursor_stats(replay_data, sample_rate, window_ms=100):
    """Precompute velocity, acceleration, and jerk magnitudes using window-sized finite differences.
    Derivatives are computed over the window interval directly, avoiding the noise amplification
    that comes from frame-by-frame differentiation at high sample rates."""
    window = max(2, int(window_ms / sample_rate))
    window_dt = window * sample_rate / 1000.0

    pos = replay_data[:, :2].copy()
    pos[:, 0] = (pos[:, 0] + 0.5) * SCREEN_WIDTH
    pos[:, 1] = (pos[:, 1] + 0.5) * SCREEN_HEIGHT

    n = len(pos)

    vel = np.zeros((n, 2))
    vel[window:] = (pos[window:] - pos[:-window]) / window_dt

    accel = np.zeros((n, 2))
    accel[window:] = (vel[window:] - vel[:-window]) / window_dt

    jerk = np.zeros((n, 2))
    jerk[window:] = (accel[window:] - accel[:-window]) / window_dt

    return np.linalg.norm(vel, axis=1), np.linalg.norm(accel, axis=1), np.linalg.norm(jerk, axis=1)


class ReplayRenderer:
    """Renders replay frames to a pygame surface at SCREEN_WIDTH x SCREEN_HEIGHT."""

    def __init__(self, beatmap, ia_replay, sample_rate, stats_window_ms=100, trail_ms=400,
                 reference_replay=None):
        self.beatmap = beatmap
        self.ia_replay = ia_replay
        self.sample_rate = sample_rate
        self.trail_ms = trail_ms
        self.preview = beatmap_preview.from_beatmap(beatmap)
        self.vel_data, self.accel_data, self.jerk_data = _compute_cursor_stats(
            ia_replay, sample_rate, stats_window_ms
        )
        self.trail = []  # list of (time_ms, px, py)
        self.reference_replay = reference_replay
        self.ref_trail = []
        self._font = None
        self._trail_surface = None

    def _ensure_initialized(self):
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", 14)
            self._trail_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self._trail_surface.set_colorkey((0, 0, 0))

    def _rebuild_trail_from(self, replay_data, time, frame):
        trail = []
        lookback = int(self.trail_ms / self.sample_rate)
        for i in range(lookback):
            f = frame - i
            if 0 < f < len(replay_data):
                tx, ty = replay_data[f, 0], replay_data[f, 1]
                t = time - i * self.sample_rate
                trail.insert(0, (t, (tx + 0.5) * SCREEN_WIDTH, (ty + 0.5) * SCREEN_HEIGHT))
            else:
                break
        return trail

    def rebuild_trail(self, time, frame):
        """Rebuild trail from current frame backwards (for seeking/scrubbing)."""
        self.trail = self._rebuild_trail_from(self.ia_replay, time, frame)
        if self.reference_replay is not None:
            self.ref_trail = self._rebuild_trail_from(self.reference_replay, time, frame)

    def render(self, surface, time, advancing=True):
        """Render one frame of the replay. Returns the replay frame index, or -1 if out of bounds."""
        self._ensure_initialized()

        surface.fill((0, 0, 0))
        self.preview.render(surface, time)

        frame = int((time - self.beatmap.start_offset()) // self.sample_rate)
        if not (0 < frame < len(self.ia_replay)):
            return -1

        # Reference replay (purple, rendered underneath)
        if self.reference_replay is not None and 0 < frame < len(self.reference_replay):
            rx, ry = self.reference_replay[frame, 0], self.reference_replay[frame, 1]
            rpx = (rx + 0.5) * SCREEN_WIDTH
            rpy = (ry + 0.5) * SCREEN_HEIGHT
            pygame.draw.circle(surface, (180, 0, 255), (int(rpx), int(rpy)), 8)

            if advancing:
                self.ref_trail.append((time, rpx, rpy))
                cutoff = time - self.trail_ms
                while self.ref_trail and self.ref_trail[0][0] < cutoff:
                    self.ref_trail.pop(0)

            if len(self.ref_trail) >= 2:
                ref_points = [(int(tx), int(ty)) for _, tx, ty in self.ref_trail]
                ref_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                for i in range(len(ref_points) - 1):
                    alpha = int(255 * (i + 1) / len(ref_points))
                    pygame.draw.line(ref_surface, (180, 0, 255, alpha),
                                     ref_points[i], ref_points[i + 1], 3)
                surface.blit(ref_surface, (0, 0))

        x, y, k1, k2 = self.ia_replay[frame]
        px = (x + 0.5) * SCREEN_WIDTH
        py = (y + 0.5) * SCREEN_HEIGHT
        pygame.draw.circle(surface, (0, 255, 0), (int(px), int(py)), 8)

        # Key indicator boxes
        box_w, box_h = 40, 30
        box_x = SCREEN_WIDTH - box_w - 10

        k1_y = SCREEN_HEIGHT // 2 - box_h - 5
        k1_color = (255, 255, 0) if k1 else (128, 128, 128)
        pygame.draw.rect(surface, k1_color, (box_x, k1_y, box_w, box_h))
        pygame.draw.rect(surface, (255, 255, 255), (box_x, k1_y, box_w, box_h), 2)

        k2_y = SCREEN_HEIGHT // 2 + 5
        k2_color = (255, 255, 0) if k2 else (128, 128, 128)
        pygame.draw.rect(surface, k2_color, (box_x, k2_y, box_w, box_h))
        pygame.draw.rect(surface, (255, 255, 255), (box_x, k2_y, box_w, box_h), 2)

        # Cursor stats overlay (top right)
        vel = self.vel_data[frame]
        acc = self.accel_data[frame]
        jrk = self.jerk_data[frame]

        stats_lines = [
            f"Vel   {vel:>8.0f} px/s",
            f"Accel {acc:>8.0f} px/s2",
            f"Jerk  {jrk:>8.0f} px/s3",
        ]

        line_h = self._font.get_height() + 2
        pad = 6
        sw = max(self._font.size(line)[0] for line in stats_lines) + pad * 2
        sh = line_h * len(stats_lines) + pad * 2
        sx = SCREEN_WIDTH - sw - 10
        sy = 10

        stats_bg = pygame.Surface((sw, sh), pygame.SRCALPHA)
        stats_bg.fill((0, 0, 0, 160))
        surface.blit(stats_bg, (sx, sy))
        pygame.draw.rect(surface, (255, 255, 255), (sx, sy, sw, sh), 1)

        for i, line in enumerate(stats_lines):
            text = self._font.render(line, True, (255, 255, 255))
            surface.blit(text, (sx + pad, sy + pad + i * line_h))

        # Trail (time-based so it looks the same regardless of render fps)
        if advancing:
            self.trail.append((time, px, py))
            cutoff = time - self.trail_ms
            while self.trail and self.trail[0][0] < cutoff:
                self.trail.pop(0)

        if len(self.trail) >= 2:
            trail_points = [(int(tx), int(ty)) for _, tx, ty in self.trail]
            trail_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            for i in range(len(trail_points) - 1):
                # Fade from transparent to opaque along the trail
                alpha = int(255 * (i + 1) / len(trail_points))
                pygame.draw.line(trail_surface, (0, 255, 0, alpha),
                                 trail_points[i], trail_points[i + 1], 3)
            surface.blit(trail_surface, (0, 0))

        return frame


def preview_replay_raw(ia_replay, beatmap_path: str, mods=None, audio_file=None,
                       sample_rate: int = REPLAY_SAMPLING_RATE, stats_window_ms: int = 100,
                       trail_ms: int = 400, reference_replay=None):
    """Interactive replay preview with playback controls (space=pause, click progress bar to seek)."""

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

    cx = 0
    cy = 0

    renderer = ReplayRenderer(beatmap, ia_replay, sample_rate, stats_window_ms, trail_ms,
                              reference_replay=reference_replay)

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
                            pygame.mixer.music.stop()
                            pygame.mixer.music.play(start=max(0, (time - beatmap['AudioLeadIn']) / 1000))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if (progress_bar_y <= mouse_y <= progress_bar_y + progress_bar_height + 10 and
                        progress_bar_margin <= mouse_x <= SCREEN_WIDTH - progress_bar_margin):
                        dragging_progress = True
                        was_paused_before_drag = paused
                        paused = True
                        if audio_file and os.path.exists(audio_file):
                            pygame.mixer.music.pause()

                        progress_ratio = (mouse_x - progress_bar_margin) / (SCREEN_WIDTH - 2 * progress_bar_margin)
                        progress_ratio = max(0, min(1, progress_ratio))
                        time = progress_start_time + progress_ratio * (progress_end_time - progress_start_time)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging_progress:
                    dragging_progress = False
                    if not was_paused_before_drag:
                        paused = False
                        if audio_file and os.path.exists(audio_file):
                            pygame.mixer.music.stop()
                            pygame.mixer.music.play(start=max(0, (time - beatmap['AudioLeadIn']) / 1000))
            elif event.type == pygame.MOUSEMOTION:
                if dragging_progress:
                    mouse_x, _ = pygame.mouse.get_pos()
                    progress_ratio = (mouse_x - progress_bar_margin) / (SCREEN_WIDTH - 2 * progress_bar_margin)
                    progress_ratio = max(0, min(1, progress_ratio))
                    time = progress_start_time + progress_ratio * (progress_end_time - progress_start_time)

        if not paused:
            time += clock.get_time()

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

        # Rebuild trail when scrubbing so it doesn't go stale
        if dragging_progress:
            frame_idx = int((time - beatmap.start_offset()) // sample_rate)
            renderer.rebuild_trail(time, frame_idx)

        advancing = not paused and not dragging_progress
        frame = renderer.render(screen, time, advancing=advancing)

        # Key press/release detection
        if frame > 0:
            _, _, k1, k2 = ia_replay[frame]
            if k1 and not prev_k1:
                print("k1 pressed")
            elif not k1 and prev_k1:
                print("k1 released")
            if k2 and not prev_k2:
                print("k2 pressed")
            elif not k2 and prev_k2:
                print("k2 released")
            prev_k1 = k1
            prev_k2 = k2

        # Progress bar
        if progress_end_time > progress_start_time:
            pygame.draw.rect(screen, (128, 128, 128),
                             (progress_bar_margin, progress_bar_y,
                              SCREEN_WIDTH - 2 * progress_bar_margin, progress_bar_height))

            progress_ratio = (time - progress_start_time) / (progress_end_time - progress_start_time)
            progress_ratio = max(0, min(1, progress_ratio))

            marker_x = progress_bar_margin + progress_ratio * (SCREEN_WIDTH - 2 * progress_bar_margin)
            pygame.draw.rect(screen, (255, 255, 255),
                             (marker_x - 2, progress_bar_y - 2, 4, progress_bar_height + 4))

        pygame.display.flip()

        clock.tick(FRAME_RATE)

    pygame.quit()


def export_replay_video(ia_replay, beatmap_path: str, output_path: str, mods=None,
                        sample_rate: int = REPLAY_SAMPLING_RATE, stats_window_ms: int = 100,
                        trail_ms: int = 400, reference_replay=None, fps: int = 60):
    """Export replay as a compressed mp4 video at SCREEN_WIDTH x SCREEN_HEIGHT. Requires ffmpeg."""

    beatmap = bm.load(beatmap_path)
    if mods:
        beatmap.apply_mods(mods)

    pygame.init()
    pygame.display.set_mode((1, 1))

    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    renderer = ReplayRenderer(beatmap, ia_replay, sample_rate, stats_window_ms, trail_ms,
                              reference_replay=reference_replay)

    start_time = beatmap.start_offset()
    end_time = start_time + len(ia_replay) * sample_rate
    frame_step = 1000.0 / fps

    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{SCREEN_WIDTH}x{SCREEN_HEIGHT}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    total_video_frames = int((end_time - start_time) / frame_step)
    t = start_time

    for _ in tqdm.tqdm(range(total_video_frames), desc="Exporting"):
        renderer.render(surface, t)
        proc.stdin.write(pygame.image.tostring(surface, 'RGB'))
        t += frame_step

    proc.stdin.close()
    proc.wait()

    pygame.quit()

    if proc.returncode != 0:
        print("ffmpeg failed - is it installed?")
    else:
        print(f"Exported to {output_path}")


def preview_replay(replay: replay_module.Replay, beatmap_path: str, audio_file=None, sampling_rate: int=dataset.SAMPLE_RATE):
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

    replay_data = dataset.target_data_single(beatmap, replay, sample_rate=sampling_rate)

    # Flatten
    ia_replay = []
    for chunk in replay_data:
        for frame in chunk:
            x, y, k1, k2 = frame[0], frame[1], frame[2], frame[3]
            ia_replay.append([x, y, k1, k2])

    ia_replay = np.array(ia_replay)

    preview_replay_raw(ia_replay, beatmap_path=beatmap_path, mods=mods, audio_file=audio_file)


# quick way to check the sanity of a dataset
def preview_training_data(xs, ys, sample_rate: int = REPLAY_SAMPLING_RATE):
    pygame.init()

    pygame.display.set_caption('Data Preview')

    time = 0
    clock = pygame.time.Clock()

    PROGRESS_BAR_HEIGHT = 50
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + PROGRESS_BAR_HEIGHT))

    FRAME_RATE = 1000

    xs_flat = xs.reshape(-1, xs.shape[-1])
    ys_flat = ys.reshape(-1, ys.shape[-1])

    total_frames = len(xs_flat)

    trail = []

    prev_k1 = False
    prev_k2 = False

    running = True
    paused = False

    # Progress bar variables
    dragging_progress = False
    was_paused_before_drag = False
    progress_start_time = 0
    progress_end_time = total_frames * sample_rate  # Total duration in ms

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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    # Check if click is on progress bar
                    if (progress_bar_y <= mouse_y <= progress_bar_y + progress_bar_height + 10 and
                        progress_bar_margin <= mouse_x <= SCREEN_WIDTH - progress_bar_margin):
                        dragging_progress = True
                        was_paused_before_drag = paused
                        paused = True

                        # Calculate new time based on click position
                        progress_ratio = (mouse_x - progress_bar_margin) / (SCREEN_WIDTH - 2 * progress_bar_margin)
                        progress_ratio = max(0, min(1, progress_ratio))
                        time = progress_ratio * progress_end_time
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging_progress:
                    dragging_progress = False
                    if not was_paused_before_drag:
                        paused = False
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
        frame = int(time // sample_rate)

        if 0 <= frame < total_frames:
            # Draw 3 green circles from xs data (current, previous, next frames)
            for offset in [-1, 0, 1]:
                target_frame = frame + offset
                if 0 <= target_frame < total_frames:
                    is_slider = xs_flat[target_frame, 3]
                    is_spinner = xs_flat[target_frame, 4]
                    is_note = xs_flat[target_frame, 5]

                    if is_slider or is_spinner or is_note:
                        # Extract x, y from xs data (first 2 columns)
                        xs_x, xs_y = xs_flat[target_frame, 0], xs_flat[target_frame, 1]

                        xs_x += 0.5
                        xs_y += 0.5
                        xs_x *= SCREEN_WIDTH
                        xs_y *= SCREEN_HEIGHT

                        # Different circle sizes for visual distinction
                        if offset == 0:  # Current frame
                            pygame.draw.circle(screen, (0, 255, 0), (int(xs_x), int(xs_y)), 12)
                        else:  # Previous/next frames
                            pygame.draw.circle(screen, (0, 200, 0), (int(xs_x), int(xs_y)), 8)

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

                # Draw key indicator boxes on the right side
                box_width = 40
                box_height = 30
                box_x = SCREEN_WIDTH - box_width - 10

                # k1 box (top)
                k1_y = SCREEN_HEIGHT // 2 - box_height - 5
                k1_color = (255, 255, 0) if k1 else (128, 128, 128)  # Yellow if pressed, grey if not
                pygame.draw.rect(screen, k1_color, (box_x, k1_y, box_width, box_height))
                pygame.draw.rect(screen, (255, 255, 255), (box_x, k1_y, box_width, box_height), 2)  # White border

                # k2 box (bottom)
                k2_y = SCREEN_HEIGHT // 2 + 5
                k2_color = (255, 255, 0) if k2 else (128, 128, 128)  # Yellow if pressed, grey if not
                pygame.draw.rect(screen, k2_color, (box_x, k2_y, box_width, box_height))
                pygame.draw.rect(screen, (255, 255, 255), (box_x, k2_y, box_width, box_height), 2)  # White border

                if dragging_progress:
                    trail = []
                    for i in range(8):
                        trail_frame = int(frame - i)
                        if trail_frame >= 0 and trail_frame < len(ys_flat):
                            tx, ty, _, _ = ys_flat[trail_frame]
                            tx += 0.5
                            ty += 0.5
                            tx *= SCREEN_WIDTH
                            ty *= SCREEN_HEIGHT
                            trail.insert(0, (tx, ty))
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
            pygame.draw.rect(screen, (128, 128, 128),
                           (progress_bar_margin, progress_bar_y,
                            SCREEN_WIDTH - 2 * progress_bar_margin, progress_bar_height))

            progress_ratio = time / progress_end_time
            progress_ratio = max(0, min(1, progress_ratio))

            marker_x = progress_bar_margin + progress_ratio * (SCREEN_WIDTH - 2 * progress_bar_margin)
            pygame.draw.rect(screen, (255, 255, 255),
                           (marker_x - 2, progress_bar_y - 2, 4, progress_bar_height + 4))

        pygame.display.flip()

        clock.tick(FRAME_RATE)

    pygame.quit()
