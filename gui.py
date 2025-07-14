import os.path

import dearpygui.dearpygui as dpg
import threading
import time
from dataclasses import dataclass
from typing import Optional
import osu.rulesets.core as osu_core
import torch
from torch import FloatTensor
import osu.dataset as dataset
import osu.rulesets.beatmap as bm
import osu.client.controller as controller
from osu.gan import OsuReplayGAN
from osu.rnn import OsuReplayRNN
from osu.rulesets.core import OSU_PATH
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tkinter import filedialog as fd


@dataclass
class CurrentBeatmapInfo:
    osu_file: str
    md5: str
    beatmap: bm.Beatmap


def do_theming_stuff():
    # Load custom font
    with dpg.font_registry():
        default_font = dpg.add_font("resources/Comfortaa-Bold.ttf", 20)

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (38, 35, 53))  # 262335
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (38, 35, 53))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (38, 35, 53))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (48, 45, 63))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (58, 55, 73))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (154, 147, 192))  # 9a93c0
            dpg.add_theme_color(dpg.mvThemeCol_Header, (36, 27, 47))  # 241b2f
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (46, 37, 57))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (56, 47, 67))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (38, 35, 53))  # title bar background
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (38, 35, 53))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (154, 147, 192))  # window border
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))  # white text on buttons
        with dpg.theme_component(dpg.mvSliderFloat):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (80, 80, 80))  # slider track
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (255, 255, 255))  # slider circle
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (220, 220, 220))
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 15)  # circle size
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 15)  # make it circular
        with dpg.theme_component(dpg.mvSliderInt):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (80, 80, 80))  # slider track
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (255, 255, 255))  # slider circle
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (220, 220, 220))
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 15)  # circle size
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 15)  # make it circular

    # Create red theme for text
    with dpg.theme(tag="red_theme"):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0, 0))
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0, 0))

    with dpg.theme(tag="green_theme"):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 255, 0))
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 255, 0))

    dpg.bind_theme(global_theme)

    return default_font


class OsuAIGUI:
    def __init__(self):
        self.map: Optional[CurrentBeatmapInfo] = None
        self.map_cache: dict[str, bm.Beatmap] = dict()
        self.map_frames_cache: dict[str, FloatTensor] = dict()
        self.play_frames_cache: dict[str, np.ndarray] = dict()
        self.osu = controller.OSUController()
        self.map_background_texture = None
        self.rnn_weights_path: Optional[str] = None
        self.gan_weights_path: Optional[str] = None
        self.rnn: Optional[OsuReplayRNN] = None
        self.gan: Optional[OsuReplayGAN] = None
        self.use_gan = True
        self.active = True

    def create_gui(self):
        dpg.create_context()

        default_font = do_theming_stuff()

        try:
            with dpg.font_registry():
                title_font = dpg.add_font("resources/Comfortaa-Bold.ttf", 26)
                artist_font = dpg.add_font("resources/Comfortaa-Bold.ttf", 10)
        except Exception as e:
            print(e)

        # TODO
        screen_height = 1080
        size = screen_height // 2

        # Main window
        with dpg.window(label="osu! AI",
                        tag="main_window",
                        width=size - 20, height=size - 40,
                        no_close=False):

            # Current beatmap section
            with dpg.group():
                dpg.add_text("Current Beatmap")
                dpg.add_separator()

                # Map preview container with background image
                with dpg.drawlist(width=size - 40, height=120, tag="map_preview"):
                    # (placeholder for map background)
                    dpg.draw_rectangle((0, 0), (size - 40, 120),
                                       fill=(50, 50, 50), color=(100, 100, 100))

                with dpg.group():
                    # Title text (larger font, bold)
                    ttext = dpg.add_text("", tag="title_text",
                                         pos=[10, 70], color=(255, 255, 255))
                    # Artist text (smaller font, bold)
                    atext = dpg.add_text("", tag="artist_text",
                                         pos=[10, 90], color=(255, 255, 255))
                    # Difficulty text (inside map preview, upper right)
                    dtext = dpg.add_text("", tag="diff_text",
                                         pos=[10, 110], color=(255, 255, 255))

            dpg.add_separator()

            # Load neural networks
            with dpg.group():
                dpg.add_text("Models")
                dpg.add_separator()
                dpg.add_button(label="Load RNN",
                               callback=self.load_rnn_weights,
                               width=140, height=30)
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Loaded weights: ")
                    dpg.add_text(f"None", tag="rnn_weights")

                dpg.add_button(label="Load GAN",
                               callback=self.load_gan_weights,
                               width=140, height=30)

                with dpg.group(horizontal=True):
                    dpg.add_text(f"Loaded weights: ")
                    dpg.add_text(f"None", tag="gan_weights")

                def toggle_gan():
                    curr = dpg.get_value("use_gan")
                    self.use_gan = curr

                dpg.add_checkbox(label="Use GAN", default_value=self.use_gan,
                                 callback=toggle_gan, tag="use_gan")

            # Control buttons
            with dpg.group():
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Generate Play",
                                   callback=self.generate_play,
                                   width=140, height=30,
                                   tag="generate_play")
                    dpg.add_button(label="Reset",
                                   callback=None,
                                   width=140, height=30)

                def set_active():
                    self.active = dpg.get_value("active")

                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Active",
                        default_value=self.active,
                        callback=set_active,
                        tag="active"
                    )

                dpg.bind_item_theme("generate_play", "red_theme")

            dpg.add_separator()

            # Settings section
            with dpg.collapsing_header(label="Settings"):
                dpg.add_checkbox(label="Auto-generate on map change",
                                 tag="auto_generate", default_value=False)
                dpg.add_slider_float(label="Playback Speed",
                                     tag="playback_speed",
                                     default_value=1.0,
                                     min_value=0.1,
                                     max_value=2.0,
                                     format="%.1fx")
                dpg.add_checkbox(label="Show Debug Info",
                                 tag="show_debug", default_value=False)
                dpg.add_slider_int(label="Smoothing Factor",
                                   tag="smoothing",
                                   default_value=5,
                                   min_value=1,
                                   max_value=20)

            dpg.add_separator()

            # Debug/Log section
            with dpg.collapsing_header(label="Debug Log"):
                dpg.add_input_text(tag="debug_log",
                                   multiline=True,
                                   readonly=True,
                                   width=size-40,
                                   height=100,
                                   default_value="Ready to start...\n")

        dpg.create_viewport(
            title="osu! AI",
            width=size, height=size,
            resizable=False
        )

        dpg.setup_dearpygui()
        dpg.bind_font(default_font)

        # font loading hates working lol
        # dpg.bind_item_font("title_text", title_font)
        # dpg.bind_item_font("artist_text", artist_font)
        # dpg.bind_item_font(dtext, artist_font)

        dpg.show_viewport()

        dpg.set_primary_window("main_window", True)

        self.refresh_beatmap()

        # Start the refresh thread
        self.start_refresh_thread()
        # Start the playing thread
        self.start_play_thread()

    def load_rnn_weights(self):
        path = fd.askopenfilename(initialdir=".")
        if path is not None and os.path.exists(path):
            try:
                self.rnn = OsuReplayRNN.load(path)
                self.rnn_weights_path = path
                dpg.set_value("rnn_weights", os.path.basename(path))
                dpg.bind_item_theme("rnn_weights", "green_theme")
            except Exception as e:
                dpg.set_value("rnn_weights", f"Invalid")
                dpg.bind_item_theme("rnn_weights", "red_theme")
                print(e)

    def load_gan_weights(self):
        path = fd.askopenfilename(initialdir=".")
        if path is not None and os.path.exists(path):
            try:
                self.gan = OsuReplayGAN.load_generator_only(path)
                self.gan_weights_path = path
                dpg.set_value("gan_weights", os.path.basename(path))
            except Exception as e:
                dpg.set_value("gan_weights", f"Invalid")
                print(e)

    def log_message(self, message: str):
        """Add message to debug log"""
        current_log = dpg.get_value("debug_log")
        timestamp = time.strftime("%H:%M:%S")
        new_log = f"{current_log}[{timestamp}] {message}\n"
        dpg.set_value("debug_log", new_log)
        print(f'[{timestamp}] {message}')

        # Auto-scroll to bottom (approximate)
        if len(new_log.split('\n')) > 10:
            lines = new_log.split('\n')
            dpg.set_value("debug_log", '\n'.join(lines[-10:]))

    def refresh_beatmap(self):
        """Refresh current beatmap info from osu! memory"""
        try:
            osu = self.osu
            status = osu.game.status()
            # 5 == song select, 2 == playing map
            if status == 5 or status == 2:
                map_md5 = osu.beatmap.md5()
                if self.map and self.map.md5 == map_md5:
                    return
                map_path = osu.beatmap.osu_file_path()
                if map_md5 not in self.map_cache:
                    self.map_cache[map_md5] = bm.load(map_path)

                self.map = CurrentBeatmapInfo(
                    osu_file=map_path,
                    beatmap=self.map_cache[map_md5],
                    md5=map_md5
                )
                self.update_beatmap_display()
                dpg.bind_item_theme(
                    "generate_play",
                    "red_theme" if self.play_frames_cache.get(map_md5) is None else "green_theme"
                )
            else:
                if self.map is not None:
                    self.map = None
                    self.update_beatmap_display()

        except Exception as e:
            pass

    def generate_play(self):
        if self.map is None:
            self.log_message(f'no map was selected')
            return

        if self.use_gan and self.gan is None:
            self.log_message(f'using the GAN to generate the play, but no weights were loaded')
            return

        if not self.use_gan and self.rnn is None:
            self.log_message(f'using the RNN to generate the play, but no weights were loaded')
            return

        # Run generation in separate thread to avoid blocking UI
        def generate_thread():
            start_md5 = self.map.md5
            if start_md5 not in self.map_frames_cache:
                data = dataset.input_data(self.map.beatmap)
                data = np.reshape(data.values, (-1, dataset.BATCH_LENGTH, len(dataset.INPUT_FEATURES)))
                data = torch.FloatTensor(data)
                self.map_frames_cache[start_md5] = data

            if self.use_gan:
                play_data = self.gan.generate(self.map_frames_cache[self.map.md5])
            else:
                play_data = self.rnn.generate(self.map_frames_cache[self.map.md5])

            play_data = np.concatenate(play_data)
            self.play_frames_cache[start_md5] = play_data
            self.log_message(f'play generated')
            dpg.bind_item_theme("generate_play", "green_theme")

        threading.Thread(target=generate_thread, daemon=True).start()

    def start_refresh_thread(self):
        """Start the background refresh thread"""
        self.refresh_running = True

        def refresh_loop():
            while self.refresh_running:
                self.refresh_beatmap()
                time.sleep(0.1)

        self.refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self.refresh_thread.start()

    def stop_refresh_thread(self):
        """Stop the background refresh thread"""
        self.refresh_running = False
        if self.refresh_thread:
            self.refresh_thread.join(timeout=0.2)

    def start_play_thread(self):
        import mouse
        def play_loop():
            # only apply movement for a frame ONCE.
            prev_frame = 0

            osu = self.osu
            while True:
                # lets not hog the cpu guys
                time.sleep(0.008)
                # check if we're in play mode
                status = osu.game.status()
                curr_md5 = osu.beatmap.md5()
                if status != 2 or not self.active or not curr_md5 == self.map.md5\
                        or self.play_frames_cache.get(curr_md5) is None:
                    # it takes more than 0.5 secs to do anything lol
                    time.sleep(0.5)
                    continue
                frames = self.play_frames_cache[curr_md5]
                curr_time = osu.game.play_time()

                REPLAY_SAMPLING_RATE = 24
                frame = int((curr_time - self.map.beatmap.start_offset()) // REPLAY_SAMPLING_RATE)
                if 0 < frame < len(frames) and prev_frame != frame:
                    prev_frame = frame
                    x, y = frames[frame]
                    x = (x + 0.5) * osu_core.SCREEN_WIDTH
                    y = (y + 0.5) * osu_core.SCREEN_HEIGHT
                    (x, y) = osu_core.osu_to_screen_pixel(x, y)
                    mouse.move(x, y)

        threading.Thread(target=play_loop, daemon=True).start()

    def update_beatmap_display(self):
        """Update the beatmap display with current map info"""
        if self.map and self.map.beatmap:
            beatmap = self.map.beatmap
            dpg.set_value("artist_text", f"{beatmap.artist()}")
            dpg.set_value("title_text", f"{beatmap.title()}")

            diff_text = beatmap.version()
            dpg.set_value("diff_text", diff_text)

            bg_name = beatmap.background_name()
            self.update_map_background(bg_name)
        else:
            dpg.set_value("artist_text", "")
            dpg.set_value("title_text", "")
            dpg.set_value("diff_text", "")

    def process_background_image(self, image_path: str):
        try:
            preview_size = (dpg.get_viewport_width(), 120)

            img = Image.open(image_path)

            aspect = img.width / float(img.height)

            new_width = preview_size[0]
            new_height = int(preview_size[0] / aspect)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.6)

            img_array = np.array(img, dtype=np.float32) / 255.0
            processed_data = img_array.flatten().tolist()

            return img.width, img.height, processed_data

        except Exception as e:
            print(f'unluncky {e}')

    def update_map_background(self, image_path: str):
        """Update the map background image"""
        if not image_path:
            dpg.delete_item("map_preview", children_only=True)
            # Show dark purple background when no image
            dpg.draw_rectangle((0, 0), (480, 120),
                               fill=(50, 50, 50), color=(100, 100, 100), parent="map_preview")
            return

        try:
            mapset_folder = os.path.dirname(self.map.osu_file)
            bg_file = os.path.join(mapset_folder, image_path)

            # Check if file exists
            if not os.path.exists(bg_file):
                dpg.delete_item("map_preview", children_only=True)
                # Show dark purple background when file not found
                dpg.draw_rectangle((0, 0), (480, 120),
                                   fill=(50, 50, 50), color=(100, 100, 100), parent="map_preview")
                return

            width, height, processed_data = self.process_background_image(bg_file)

            if not processed_data or width <= 0 or height <= 0:
                dpg.delete_item("map_preview", children_only=True)
                # dark purple
                dpg.draw_rectangle((0, 0), (480, 120),
                                   fill=(50, 50, 50), color=(100, 100, 100), parent="map_preview")
                return

            with dpg.texture_registry():
                new_texture = dpg.add_static_texture(
                    width=width, height=height, default_value=processed_data
                )

            # Calculate scaled size to fit preview area
            preview_width = dpg.get_viewport_width()
            preview_height = 120

            aspect_ratio = width / height
            if width > height:
                scaled_width = preview_width
                scaled_height = preview_width / aspect_ratio
            else:
                scaled_height = preview_height
                scaled_width = preview_height * aspect_ratio

            # Center the image
            x_offset = (preview_width - scaled_width) / 2
            y_offset = (preview_height - scaled_height) / 2

            dpg.delete_item("map_preview", children_only=True)
            dpg.draw_image(new_texture,
                           (x_offset, y_offset),
                           (x_offset + scaled_width, y_offset + scaled_height),
                           parent="map_preview")

            # Clean up old texture after new one is displayed
            if self.map_background_texture:
                dpg.delete_item(self.map_background_texture)
            self.map_background_texture = new_texture

        except Exception as e:
            dpg.delete_item("map_preview", children_only=True)
            # Show dark purple background on any error
            dpg.draw_rectangle((0, 0), (480, 120),
                               fill=(50, 50, 50), color=(100, 100, 100), parent="map_preview")

    def run(self):
        try:
            dpg.start_dearpygui()
        except KeyboardInterrupt:
            self.log_message("Shutting down...")
        finally:
            self.stop_refresh_thread()
            dpg.destroy_context()


def main():
    gui = OsuAIGUI()
    gui.create_gui()
    gui.run()


if __name__ == "__main__":
    main()
