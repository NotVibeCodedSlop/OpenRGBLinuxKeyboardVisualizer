import subprocess
import numpy as np
import os
import fcntl
import threading
import time
import random
from collections import deque
from openrgb import OpenRGBClient
from openrgb.utils import RGBColor

try:
    import evdev
except ImportError:
    print("Error: 'evdev' library is required for keystroke listening on Wayland.")
    print("Please install it using: pip install evdev")
    exit(1)

# --- CONFIGURATION (YOUR PRESERVED CHANGES) ---
KEYBOARD_NAME = "SteelSeries Apex 3 TKL"
TARGET_FPS = 120
DECAY_RATE = 0.1  # How fast the keypress green effect fades
MIN_BEAT_PEAK = 1000  # Noise floor to prevent triggering on silence/hiss

# --- BEAT DETECTION CONFIGURATION ---
BEAT_DURATION = 0.05           # How long the blue flash lasts in seconds

# Sensitive Mode Settings (Spectral Flux)
BEAT_THRESHOLD = 2.5         # Spike threshold over average

# Bass Mode Settings (Sub-125Hz Energy)
BASS_CUTOFF_HZ = 125          # Frequencies below this are considered bass
BASS_THRESHOLD = 1.50         # Spike threshold over average

# --- NEW CONFIGURATION VARIABLES ---
RANDOM_COLORS = True          # If True, colors randomize on every beat. If False, uses default Orange/Blue.
CONTRAST_THRESHOLD = 0.75     # Minimum contrast between randomized colors (0.25 = 25% contrast)
# -------------------------------------

# Map keys to their physical horizontal LED segments (0 to 7)
KEY_TO_ZONE = {
    # Zone 0 (Far Left)
    'esc': 0, '`': 0, '1': 0, 'tab': 0, 'q': 0, 'caps_lock': 0, 'capslock': 0, 'a': 0, 'shift': 0, 'leftshift': 0, 'z': 0, 'ctrl': 0, 'leftctrl': 0, 'alt': 0, 'leftalt': 0, 'f1': 0, 'f2': 0,
    # Zone 1
    '2': 1, '3': 1, 'w': 1, 's': 1, 'x': 1, 'f3': 1, 'f4': 1,
    # Zone 2
    '4': 2, 'e': 2, 'd': 2, 'c': 2, 'f5': 2,
    # Zone 3
    '5': 3, '6': 3, 'r': 3, 't': 3, 'f': 3, 'g': 3, 'v': 3, 'b': 3, 'space': 3, 'f6': 3, 'f7': 3,
    # Zone 4
    '7': 4, '8': 4, 'y': 4, 'u': 4, 'h': 4, 'j': 4, 'n': 4, 'm': 4, 'f8': 4, 'f9': 4,
    # Zone 5
    '9': 5, '0': 5, 'i': 5, 'o': 5, 'k': 5, 'l': 5, ',': 5, '.': 5, 'f10': 5,
    # Zone 6
    '-': 6, '=': 6, 'p': 6, '[': 6, ']': 6, ';': 6, "'": 6, '/': 6, 'enter': 6, 'backspace': 6, '\\': 6, 'f11': 6, 'f12': 6, 'rightshift': 6, 'rightctrl': 6, 'rightalt': 6,
    # Zone 7 (Far Right)
    'insert': 7, 'delete': 7, 'home': 7, 'end': 7, 'page_up': 7, 'page_down': 7, 'pageup': 7, 'pagedown': 7,
    'left': 7, 'up': 7, 'down': 7, 'right': 7, 'print_screen': 7, 'printscreen': 7, 'scroll_lock': 7, 'scrolllock': 7, 'pause': 7
}

def get_luminance(color):
    """Calculates relative luminance using ITU-R BT.601 luma formula."""
    return 0.299 * color.red + 0.587 * color.green + 0.114 * color.blue

def generate_random_color():
    """Generates a completely random RGBColor."""
    return RGBColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_color_with_contrast(ref_color, min_contrast=64):
    """Generates a random color with at least min_contrast luma diff from ref_color."""
    ref_lum = get_luminance(ref_color)
    for _ in range(100):  # Attempt up to 100 times to find a high-contrast match
        color = generate_random_color()
        if abs(get_luminance(color) - ref_lum) >= min_contrast:
            return color
    # Fallback to inverted color if no match is found
    return RGBColor(255 - ref_color.red, 255 - ref_color.green, 255 - ref_color.blue)

def find_keyboard_devices():
    """Finds all input devices matching the keyboard name that support key events."""
    try:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    except PermissionError:
        print("\n[Error] Permission denied when accessing input devices.")
        print("To fix this, either:")
        print("  1. Run this script with sudo: sudo python <script_name>.py")
        print("  2. Add your user to the 'input' group: sudo usermod -aG input $USER")
        print("     (Note: You must log out and log back in for group changes to take effect)\n")
        return []

    matched_devices = []
    for dev in devices:
        if KEYBOARD_NAME in dev.name:
            caps = dev.capabilities()
            if evdev.ecodes.EV_KEY in caps:
                matched_devices.append(dev)

    if not matched_devices:
        for dev in devices:
            if "keyboard" in dev.name.lower() or "kbd" in dev.name.lower():
                caps = dev.capabilities()
                if evdev.ecodes.EV_KEY in caps:
                    matched_devices.append(dev)

    return matched_devices

def interpolate_color(c1, c2, factor):
    """Linearly interpolates between two RGBColor objects."""
    r = int(c1.red * factor + c2.red * (1.0 - factor))
    g = int(c1.green * factor + c2.green * (1.0 - factor))
    b = int(c1.blue * factor + c2.blue * (1.0 - factor))
    return RGBColor(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

class KeyboardThread(threading.Thread):
    def __init__(self, kb, target_fps=60):
        super().__init__(daemon=True)
        self.kb = kb
        self.target_interval = 1.0 / target_fps
        self.audio_levels = [0] * 8
        self.key_effects = [0.0] * 8  # 0.0 (no effect) to 1.0 (full green)
        self.beat_intensity = 0.0  # 1.0 (full beat gradient) decaying to 0.0 (idle gradient)
        self.event = threading.Event()
        self.lock = threading.Lock()

        # Default static gradient colors (used if RANDOM_COLORS is False)
        self.idle_left = RGBColor(255, 50, 0)    # Default Orange
        self.idle_right = RGBColor(255, 150, 0)  # Default Light Orange
        self.beat_left = RGBColor(0, 100, 255)   # Default Blue
        self.beat_right = RGBColor(0, 200, 255)  # Default Light Blue

        if RANDOM_COLORS:
            self.randomize_colors()

    def randomize_colors(self):
        """Generates new random gradient endpoints with guaranteed contrast."""
        min_contrast = int(CONTRAST_THRESHOLD * 255)
        self.idle_left = generate_random_color()
        # Ensure the idle gradient itself has visible contrast
        self.idle_right = generate_color_with_contrast(self.idle_left, min_contrast=min_contrast)
        # Ensure the beat gradient has contrast against the idle gradient
        self.beat_left = generate_color_with_contrast(self.idle_left, min_contrast=min_contrast)
        self.beat_right = generate_color_with_contrast(self.idle_right, min_contrast=min_contrast)

    def update_audio_levels(self, levels, is_beat=False):
        with self.lock:
            self.audio_levels = levels
            if is_beat:
                self.beat_intensity = 1.0  # Reset beat flash to maximum intensity
                if RANDOM_COLORS:
                    self.randomize_colors()    # Morph to a new high-contrast color palette
        self.event.set()

    def trigger_key_effect(self, zone):
        with self.lock:
            self.key_effects[zone] = 1.0
        self.event.set()

    def run(self):
        last_update_time = time.perf_counter()
        GREEN = RGBColor(0, 255, 0)
        ORANGE = RGBColor(255, 50, 0)

        # Calculate how much the beat intensity decays per frame
        beat_decay = self.target_interval / BEAT_DURATION if BEAT_DURATION > 0 else 1.0

        while True:
            self.event.wait()
            self.event.clear()

            # Rate limiting to target FPS
            now = time.perf_counter()
            elapsed = now - last_update_time
            if elapsed < self.target_interval:
                time.sleep(self.target_interval - elapsed)
                now = time.perf_counter()

            actual_fps = 1.0 / (now - last_update_time)
            last_update_time = now

            with self.lock:
                audio_levels = list(self.audio_levels)
                key_effects = list(self.key_effects)
                beat_intensity = self.beat_intensity

                # Copy gradient colors locally to prevent race conditions during updates
                idle_left = self.idle_left
                idle_right = self.idle_right
                beat_left = self.beat_left
                beat_right = self.beat_right

                # Decay the keypress effects for the next frame
                for i in range(8):
                    if self.key_effects[i] > 0.0:
                        self.key_effects[i] = max(0.0, self.key_effects[i] - DECAY_RATE)

                # Decay the beat flash intensity for the next frame
                if self.beat_intensity > 0.0:
                    self.beat_intensity = max(0.0, self.beat_intensity - beat_decay)

            # Calculate final colors for each of the 8 zones
            final_colors = []
            for i in range(8):
                factor_zone = i / 7.0  # Position across the 8 zones (0.0 to 1.0)

                # Interpolate the idle and beat gradient colors for this specific zone
                idle_color_i = interpolate_color(idle_right, idle_left, factor_zone)
                beat_color_i = interpolate_color(beat_right, beat_left, factor_zone)

                # Interpolate between the idle and beat gradients based on current beat intensity
                active_bg_color = interpolate_color(beat_color_i, idle_color_i, beat_intensity)

                # Base background color from the audio visualizer
                bg_color = active_bg_color if audio_levels[i] > 0 else RGBColor(0, 0, 0)

                t = key_effects[i]
                if t > 0.5:
                    # First half of fade: Green -> Orange
                    factor = (t - 0.5) / 0.5
                    color = interpolate_color(GREEN, ORANGE, factor)
                elif t > 0.0:
                    # Second half of fade: Orange -> Background (Audio/Black)
                    factor = t / 0.5
                    color = interpolate_color(ORANGE, bg_color, factor)
                else:
                    color = bg_color

                final_colors.append(color)

            try:
                start = time.perf_counter()
                self.kb.set_colors(final_colors, fast=True)
                usb_time = (time.perf_counter() - start) * 1000
                print(f"FPS: {actual_fps:.1f} | USB: {usb_time:.2f}ms", end="\r")
            except Exception:
                pass

class EvdevListenerThread(threading.Thread):
    def __init__(self, dev, kb_thread):
        super().__init__(daemon=True)
        self.dev = dev
        self.kb_thread = kb_thread

    def run(self):
        print(f"Listening to keystrokes on: {self.dev.name} ({self.dev.path})")
        try:
            for event in self.dev.read_loop():
                if event.type == evdev.ecodes.EV_KEY:
                    key_event = evdev.categorize(event)
                    # Only trigger on key down (press), not key up (release) or key hold
                    if key_event.keystate == evdev.events.KeyEvent.key_down:
                        keycode = key_event.keycode
                        if isinstance(keycode, list):
                            keycode = keycode[0]

                        key_name = keycode.lower()
                        if key_name.startswith("key_"):
                            key_name = key_name[4:]

                        zone = KEY_TO_ZONE.get(key_name)
                        if zone is None:
                            # Fallback: hash the key name to a zone
                            zone = hash(key_name) % 8

                        self.kb_thread.trigger_key_effect(zone)
        except Exception as e:
            print(f"[Warning] Keystroke listener error on {self.dev.path}: {e}")

def main():
    print(f"Connecting to OpenRGB and finding {KEYBOARD_NAME}...")
    client = OpenRGBClient(address='127.0.0.1', port=6742)
    kb = client.get_devices_by_name(KEYBOARD_NAME)[0]
    kb.set_mode('direct')

    kb_thread = KeyboardThread(kb, target_fps=TARGET_FPS)
    kb_thread.start()

    # Start the evdev keystroke listener threads for ALL matching devices
    devices = find_keyboard_devices()
    if not devices:
        print("[Warning] No keyboard devices found for evdev. Keystroke effects disabled.")
    else:
        for dev in devices:
            listener_thread = EvdevListenerThread(dev, kb_thread)
            listener_thread.start()

    # Start parec with low latency environment variable and flag
    env = os.environ.copy()
    env['PULSE_LATENCY_MSEC'] = '16'

    cmd = [
        'parec',
        '--device', '@DEFAULT_MONITOR@',
        '--format=s16le',
        '--rate=44100',
        '--channels=1',
        '--latency-msec=16'
    ]
    # bufsize=0 disables pipe buffering so we get raw audio chunks instantly
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, bufsize=0)

    # Set the pipe to blocking mode.
    # Reading exactly 1024 bytes (512 samples) will block for exactly 11.6ms,
    # acting as a natural, zero-CPU sleep.
    print(f"Visualizer started targeting {TARGET_FPS}Hz. Press Ctrl+C to stop.")

    last_audio_time = time.perf_counter()

    # Parallel Beat detection variables
    prev_magnitude = None
    flux_history = deque(maxlen=50)  # ~500ms of history at 86Hz frame rate
    bass_history = deque(maxlen=50)  # ~500ms of history for bass tracking

    cooldown_sensitive = 0
    cooldown_bass = 0

    last_flux = 0
    last_bass = 0

    # Calculate the maximum frequency bin index for the bass cutoff.
    # We use a 1024-point zero-padded FFT for high-resolution bass tracking.
    bin_resolution = 44100 / 1024  # ~43.07 Hz per bin
    max_bass_bin = max(1, int(BASS_CUTOFF_HZ / bin_resolution))

    try:
        while True:
            # Read exactly 1024 bytes (512 samples of 16-bit mono audio)
            raw_data = process.stdout.read(1024)
            if not raw_data:
                break

            data = np.frombuffer(raw_data, dtype=np.int16)
            if data.size < 512:
                continue

            # 1. Compute volume level for the visualizer (PRESERVED SCALE CHANGE)
            peak = np.max(np.abs(data))
            level = min(int(peak / 20000 * 8), 8)

            # 2. Compute FFT with zero-padding to 1024 points for high-resolution bass tracking
            window = np.hanning(len(data))
            fft_data = np.fft.rfft(data * window, n=1024)
            magnitude = np.abs(fft_data)

            is_beat = False

            # --- BASS BEAT DETECTION ---
            # Sum the energy of all frequency bins below BASS_CUTOFF_HZ (excluding DC offset at bin 0)
            bass_energy = np.sum(magnitude[1:max_bass_bin + 1])
            bass_history.append(bass_energy)

            if len(bass_history) >= 15:
                avg_bass = sum(bass_history) / len(bass_history)

                # Trigger beat if bass energy spikes above the rolling average threshold
                if (peak > MIN_BEAT_PEAK and
                    bass_energy > avg_bass * BASS_THRESHOLD and
                    bass_energy > last_bass and
                    cooldown_bass == 0):
                    is_beat = True
                    cooldown_bass = 6  # ~70ms cooldown for bass transients

            last_bass = bass_energy

            # --- SENSITIVE BEAT DETECTION (Spectral Flux) ---
            if prev_magnitude is not None:
                # Compute positive difference (onset of energy across all frequencies)
                diff = magnitude - prev_magnitude
                flux = np.sum(np.maximum(0, diff))
                flux_history.append(flux)

                if len(flux_history) >= 15:
                    avg_flux = sum(flux_history) / len(flux_history)

                    # Trigger beat if spectral flux spikes above the rolling average threshold
                    if (peak > MIN_BEAT_PEAK and
                        flux > avg_flux * BEAT_THRESHOLD and
                        flux > last_flux and
                        cooldown_sensitive == 0):
                        is_beat = True
                        cooldown_sensitive = 6  # ~70ms cooldown for high-frequency transients

                last_flux = flux

            prev_magnitude = magnitude

            # Decrement independent cooldown timers
            if cooldown_bass > 0:
                cooldown_bass -= 1
            if cooldown_sensitive > 0:
                cooldown_sensitive -= 1

            # Map single level to 8 zones
            audio_levels = [1 if i < level else 0 for i in range(8)]
            kb_thread.update_audio_levels(audio_levels, is_beat)

            if level > 0:
                last_audio_time = time.perf_counter()
            else:
                # If there has been no audio for more than 100ms, fade to 0
                if time.perf_counter() - last_audio_time > 0.1:
                    kb_thread.update_audio_levels([0] * 8)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        process.terminate()
        try:
            kb.set_colors([RGBColor(0, 0, 0)] * 8, fast=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()
