import dearpygui.dearpygui as dpg
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class BeatmapInfo:
    artist: str
    title: str
    difficulty: str
    beatmap_id: int


def do_theming_stuff():
    # Load custom font
    with dpg.font_registry():
        default_font = dpg.add_font("resources/Comfortaa-Bold.ttf", 20)

    # Configure theme for better appearance
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

    dpg.bind_theme(global_theme)

    return default_font

class OsuAIGUI:
    def __init__(self):
        self.current_beatmap: Optional[BeatmapInfo] = None
        self.is_generating = False
        self.is_playing = False
        self.replay_data = None
        self.map_background_texture = None

    def create_gui(self):
        dpg.create_context()

        default_font = do_theming_stuff()

        # TODO
        screen_height = 1080
        size = screen_height // 2

        # Main window
        with dpg.window(label="osu! AI Assistant",
                        tag="main_window",
                        width=size-20, height=size-40,
                        no_close=False):
            # Current beatmap section - map preview with background
            with dpg.group():
                dpg.add_text("Current Beatmap")
                dpg.add_separator()
                
                # Map preview container with background image
                with dpg.drawlist(width=size-40, height=120, tag="map_preview"):
                    # Background rectangle (placeholder for map background)
                    dpg.draw_rectangle((0, 0), (size-40, 120), 
                                     fill=(50, 50, 50), color=(100, 100, 100))
                    
                    # Semi-transparent overlay for text readability
                    dpg.draw_rectangle((0, 80), (size-40, 120), 
                                     fill=(0, 0, 0, 180), color=(0, 0, 0, 0))
                
                # Text overlay on the preview (positioned over the drawlist)
                with dpg.group():
                    dpg.add_text("Artist: Not detected", tag="artist_text", 
                               pos=[10, 90], color=(255, 255, 255))
                    dpg.add_text("Title: Not detected", tag="title_text", 
                               pos=[10, 105], color=(255, 255, 255))
                    dpg.add_text("Difficulty: Not detected", tag="diff_text", 
                               pos=[10, 120], color=(255, 255, 255))
                
                dpg.add_button(label="Refresh Beatmap Info", callback=self.refresh_beatmap)

            dpg.add_separator()

            # AI Status section
            with dpg.group():
                dpg.add_text("AI Status")
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_text("Status:")
                    dpg.add_text("Ready", tag="status_text", color=(0, 255, 0))

                with dpg.group(horizontal=True):
                    dpg.add_text("Replay Data:")
                    dpg.add_text("None", tag="replay_status", color=(255, 100, 100))

            dpg.add_separator()

            # Control buttons
            with dpg.group():
                dpg.add_text("Controls")
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Generate Replay",
                                   callback=self.generate_replay,
                                   width=140, height=30)
                    dpg.add_button(label="Start AI Play",
                                   callback=self.start_ai_play,
                                   width=140, height=30,
                                   tag="play_button")

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Stop AI",
                                   callback=self.stop_ai,
                                   width=140, height=30)
                    dpg.add_button(label="Reset",
                                   callback=self.reset_ai,
                                   width=140, height=30)

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
                                   height=100,
                                   default_value="Ready to start...\n")

        dpg.create_viewport(
            title="osu! AI",
            width=size, height=size,
            resizable=False
        )

        dpg.setup_dearpygui()
        dpg.bind_font(default_font)
        dpg.show_viewport()

        dpg.set_primary_window("main_window", True)

    def log_message(self, message: str):
        """Add message to debug log"""
        current_log = dpg.get_value("debug_log")
        timestamp = time.strftime("%H:%M:%S")
        new_log = f"{current_log}[{timestamp}] {message}\n"
        dpg.set_value("debug_log", new_log)

        # Auto-scroll to bottom (approximate)
        if len(new_log.split('\n')) > 10:
            lines = new_log.split('\n')
            dpg.set_value("debug_log", '\n'.join(lines[-10:]))

    def refresh_beatmap(self):
        """Refresh current beatmap info from osu! memory"""
        self.log_message("Refreshing beatmap info...")

        # TODO: Replace with actual memory reading
        # For now, simulate beatmap detection
        try:
            # This is where you'd call your OsuMemoryReader
            # beatmap = memory_reader.get_current_beatmap()

            # Simulated data for testing
            self.current_beatmap = BeatmapInfo(
                artist="Example Artist",
                title="Example Song",
                difficulty="Hard",
                beatmap_id=12345
            )

            if self.current_beatmap:
                dpg.set_value("artist_text", f"Artist: {self.current_beatmap.artist}")
                dpg.set_value("title_text", f"Title: {self.current_beatmap.title}")
                dpg.set_value("diff_text", f"Difficulty: {self.current_beatmap.difficulty}")
                self.log_message(f"Detected: {self.current_beatmap.artist} - {self.current_beatmap.title}")

                # Auto-generate if enabled
                if dpg.get_value("auto_generate"):
                    self.generate_replay()
            else:
                self.log_message("No beatmap detected")

        except Exception as e:
            self.log_message(f"Error reading beatmap: {str(e)}")

    def generate_replay(self):
        """Generate AI replay for current beatmap"""
        if self.is_generating:
            self.log_message("Already generating replay...")
            return

        if not self.current_beatmap:
            self.log_message("No beatmap selected. Please refresh beatmap info first.")
            return

        self.is_generating = True
        dpg.set_value("status_text", "Generating...")
        dpg.configure_item("status_text", color=(255, 255, 0))
        self.log_message(f"Generating replay for {self.current_beatmap.title}...")

        # Run generation in separate thread to avoid blocking UI
        def generate_thread():
            try:
                # TODO: Replace with actual GAN inference
                # self.replay_data = replay_generator.generate_replay(self.current_beatmap)

                # Simulate generation time
                time.sleep(2)

                # Simulated replay data
                self.replay_data = [(i * 100, 200 + i, 300 + i, False) for i in range(100)]

                # Update UI on main thread
                dpg.set_value("status_text", "Ready")
                dpg.configure_item("status_text", color=(0, 255, 0))
                dpg.set_value("replay_status", "Generated")
                dpg.configure_item("replay_status", color=(0, 255, 0))
                self.log_message("Replay generation complete!")

            except Exception as e:
                dpg.set_value("status_text", "Error")
                dpg.configure_item("status_text", color=(255, 0, 0))
                self.log_message(f"Generation error: {str(e)}")

            finally:
                self.is_generating = False

        threading.Thread(target=generate_thread, daemon=True).start()

    def start_ai_play(self):
        """Start AI playback"""
        if not self.replay_data:
            self.log_message("No replay data available. Generate a replay first.")
            return

        if self.is_playing:
            self.log_message("AI is already playing...")
            return

        self.is_playing = True
        dpg.set_value("status_text", "Playing")
        dpg.configure_item("status_text", color=(0, 255, 255))
        dpg.configure_item("play_button", label="Playing...")

        speed = dpg.get_value("playback_speed")
        self.log_message(f"Starting AI playback at {speed}x speed...")

        # TODO: Replace with actual mouse controller
        def play_thread():
            try:
                # This is where you'd call your PreciseMouseController
                # mouse_controller.start_replay_execution(self.replay_data, memory_reader)

                # Simulate playback
                for i, (timestamp, x, y, keys) in enumerate(self.replay_data):
                    if not self.is_playing:
                        break
                    time.sleep(0.01 / speed)  # Simulated timing

                    if dpg.get_value("show_debug") and i % 10 == 0:
                        self.log_message(f"Position: ({x}, {y}) at {timestamp}ms")

                self.log_message("Playback completed!")

            except Exception as e:
                self.log_message(f"Playback error: {str(e)}")
            finally:
                self.is_playing = False
                dpg.set_value("status_text", "Ready")
                dpg.configure_item("status_text", color=(0, 255, 0))
                dpg.configure_item("play_button", label="Start AI Play")

        threading.Thread(target=play_thread, daemon=True).start()

    def stop_ai(self):
        """Stop AI playback"""
        if self.is_playing:
            self.is_playing = False
            self.log_message("AI playback stopped")
            dpg.set_value("status_text", "Stopped")
            dpg.configure_item("status_text", color=(255, 0, 0))
            dpg.configure_item("play_button", label="Start AI Play")
        else:
            self.log_message("AI is not currently playing")

    def reset_ai(self):
        """Reset AI state"""
        self.stop_ai()
        self.replay_data = None
        self.current_beatmap = None

        # Reset UI
        dpg.set_value("artist_text", "Artist: Not detected")
        dpg.set_value("title_text", "Title: Not detected")
        dpg.set_value("diff_text", "Difficulty: Not detected")
        dpg.set_value("replay_status", "None")
        dpg.configure_item("replay_status", color=(255, 100, 100))
        dpg.set_value("status_text", "Ready")
        dpg.configure_item("status_text", color=(0, 255, 0))
        dpg.set_value("debug_log", "Reset complete...\n")

        self.log_message("AI state reset")

    def update_map_background(self, image_path: str):
        """Update the map background image"""
        try:
            # Load texture from image file
            width, height, channels, data = dpg.load_image(image_path)
            
            # Create texture if it doesn't exist
            if self.map_background_texture:
                dpg.delete_item(self.map_background_texture)
            
            self.map_background_texture = dpg.add_static_texture(
                width=width, height=height, default_value=data
            )
            
            # Clear the drawlist and redraw with new background
            dpg.delete_item("map_preview", children_only=True)
            
            # Draw the background image
            dpg.draw_image(self.map_background_texture, (0, 0), (width, height), parent="map_preview")
            
            # Redraw the overlay
            dpg.draw_rectangle((0, 80), (width, 120), 
                             fill=(0, 0, 0, 180), color=(0, 0, 0, 0), parent="map_preview")
            
            self.log_message(f"Updated map background: {image_path}")
            
        except Exception as e:
            self.log_message(f"Failed to load background image: {str(e)}")

    def run(self):
        """Start the GUI"""
        try:
            dpg.start_dearpygui()
        except KeyboardInterrupt:
            self.log_message("Shutting down...")
        finally:
            dpg.destroy_context()


def main():
    """Main entry point"""
    gui = OsuAIGUI()
    gui.create_gui()
    gui.run()


if __name__ == "__main__":
    # main()

    import osu.client.controller as controller
    osu = controller.OSUController()
    while True:
        print(osu.game.status())
