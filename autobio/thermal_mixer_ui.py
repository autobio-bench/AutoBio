from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors
import matplotlib.image
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
import numpy as np

ASSET_ROOT = Path(__file__).parent / "assets"
FONT_ROOT = ASSET_ROOT / "common" / "font"
ICON_ROOT = ASSET_ROOT / "instrument" / "thermal_mixer_eppendorf_c" / "icon"

matplotlib.font_manager.fontManager.addfont(FONT_ROOT / 'NotoSans-Bold.ttf')
matplotlib.font_manager.fontManager.addfont(FONT_ROOT / 'NotoSansMath-Regular.ttf')

# --- UI Configuration ---
# Aspect of 0.098737 / 0.025772
# Approximate layout: 15 + 1 + 18 + 13 + 2 + 4 + 7 + 1

WIDTH = 249  # Canvas width in pixels AND points
HEIGHT = 65  # Canvas height in pixels AND points
RESOLUTION = 72  # DPI

# Colors
BG_COLOR = "white"
TEXT_COLOR = "black"
BORDER_COLOR = "black"
HIGHLIGHT_COLOR = "blue"  # For active states or selections (placeholder)

# Fonts
DEFAULT_FONT = "Noto Sans"
LARGE_FONT_SIZE = 18
MEDIUM_FONT_SIZE = 12
SMALL_FONT_SIZE = 10

# Layout Coordinates

# Status Bar
STATUS_BAR_HEIGHT = 15
STATUS_BAR_Y = HEIGHT - STATUS_BAR_HEIGHT
PROGRAM_INFO_X = 5
ICON_START_X = WIDTH  # Start drawing icons from this x-coordinate (RTL)
ICON_Y = STATUS_BAR_Y + STATUS_BAR_HEIGHT / 2
ICON_SPACING = STATUS_BAR_HEIGHT + 1

# Main Parameters
MAIN_PARAM_Y = 0
TIME_X = WIDTH * 0.15
TEMP_X = WIDTH * 0.5
FREQ_X = WIDTH * 0.85
PARAM_UNIT_Y = MAIN_PARAM_Y + 3
PARAM_UNIT_HEIGHT = 8
PARAM_VALUE_Y = PARAM_UNIT_Y + PARAM_UNIT_HEIGHT + 3
PARAM_VALUE_HEIGHT = 14

# ICONS
ICON_PATHS = {
    "running": "running.png",
    "paused": "paused.png",
    "interval_mix_on": "interval_mix.png",
    "speaker_on": "speaker_on.png",
    "speaker_off": "speaker_off.png",  # Usually indicated by absence of 'speaker_on'
    "key_lock_on": "lock.png",
    "key_lock_off": "unlock.png",
    "time_control": "time_control.png",
    "temp_control": "temp_control.png",
}
ICONS = {key: matplotlib.image.imread(ICON_ROOT / path) for key, path in ICON_PATHS.items()}
ICON_CMAP = matplotlib.colors.ListedColormap(
    [
        (0, 0, 0, 1),  # Black (opaque)
        (1, 1, 1, 0),  # White (transparent)
    ],
    name="binary",
    N=2,
)

def draw_text(ax: plt.Axes, x: float, y: float, text: str, fontsize: int, *, ha="center", fontfamily=DEFAULT_FONT, height=None):
    if height is None:
        height = fontsize
    ax.text(
        x, y + height / 2, text,
        ha=ha, va="center",
        fontsize=fontsize,
        color=TEXT_COLOR,
        fontfamily=fontfamily,
        weight="bold",
        antialiased=True,
        # bbox={'edgecolor': 'black', 'facecolor': 'None', 'boxstyle': 'round,pad=0.0'},
    )


@dataclass
class Time:
    seconds: int

    @property
    def is_infinite(self) -> bool:
        return self.seconds >= 360000

    @property
    def step_size(self) -> int:
        if self.seconds < 5 * 60:
            return 5
        elif self.seconds < 20 * 60:
            return 15
        elif self.seconds < 60 * 60:
            return 60
        elif self.seconds < 10 * 60 * 60:
            return 300
        else:
            return 1800

    def format(self) -> tuple[str, str]:
        if self.is_infinite:
            return "∞", ""
        hours = self.seconds // 3600
        minutes = (self.seconds % 3600) // 60
        seconds = self.seconds % 60
        if hours > 0:
            time = f"{hours}:{minutes:02d}"
            unit = "h : min"
        else:
            time = f"{minutes}:{seconds:02d}"
            unit = "min : s"
        return time, unit


@dataclass
class StatusBar:
    """
    Status bar.
    Displays the current program number and name (if loaded), and status icons (RTL).
    """

    # LTR
    program_number: int | None  # 1
    program_name: str | None  # 2

    # RTL
    time_mode: str  # 9 ('time_control' or 'temp_control')
    key_lock: bool  # 8
    speaker: bool  # 7 (True for on, False for off - icon usually shown only for ON)
    interval_mix: bool  # 6
    thermotop: None  # 5 (Unsupported)
    device_status: str | None  # 4 ('running' or 'paused')

    def draw(self, ax: plt.Axes):
        """Draws the status bar onto the axes."""
        # Optional: Clear previous content if redrawing dynamically
        # ax.add_patch(patches.Rectangle((0, STATUS_BAR_Y), WIDTH, STATUS_BAR_HEIGHT, facecolor=BG_COLOR, edgecolor=None))

        # Draw top border line (as seen in image)
        ax.plot(
            [0, WIDTH], [STATUS_BAR_Y, STATUS_BAR_Y], color=BORDER_COLOR, linewidth=1
        )

        # 1 & 2: Draw Program Number and Name (LTR)
        prog_text = ""
        if self.program_number is not None:
            prog_text += f"P {self.program_number:02d}"  # Pad with zero if needed
        if self.program_name:
            prog_text += f" {self.program_name}"

        if prog_text:
            draw_text(ax, PROGRAM_INFO_X, ICON_Y, prog_text, MEDIUM_FONT_SIZE, ha="left")

        # Draw Status Icons (RTL)
        current_icon_x = ICON_START_X

        def add_icon(icon_key: str | None):
            nonlocal current_icon_x
            if icon_key:
                icon = ICONS[icon_key]
                height, width = icon.shape
                extent = (
                    current_icon_x - ICON_SPACING / 2 - width / 2,
                    current_icon_x - ICON_SPACING / 2 + width / 2,
                    ICON_Y - height / 2,
                    ICON_Y + height / 2,
                )
                # white pixel as transparent
                ax.imshow(icon, extent=extent, cmap=ICON_CMAP, vmin=0, vmax=1)

            current_icon_x -= ICON_SPACING
            ax.plot(
                [current_icon_x, current_icon_x],
                [STATUS_BAR_Y, STATUS_BAR_Y + STATUS_BAR_HEIGHT],
                color=BORDER_COLOR,
                linewidth=1,
            )

        add_icon(self.time_mode)
        add_icon("key_lock_on" if self.key_lock else "key_lock_off")
        add_icon("speaker_on" if self.speaker else "speaker_off")
        add_icon("interval_mix_on" if self.interval_mix else None)
        add_icon(self.thermotop)
        add_icon(self.device_status)


@dataclass
class MainParameter:
    """
    Main parameter row.
    Displays the current values of the main parameters (time, temperature [set/actual], frequency).
    """

    time: Time  # seconds, >= 360000 (100h) means time is continuous, 13
    time_pause: bool
    set_temperature: float  # celsius, 0 means temperature is off, 12
    actual_temperature: float  # celsius, 11
    frequency: int  # rpm, 0 means mixing is off, 10

    def draw(self, ax: plt.Axes):
        """Draws the main parameters onto the axes."""

        # 13: Time
        time_str, time_unit_str = self.time.format()
        if self.time_pause:
            time_str = "Pause"
        if self.time.is_infinite:
            draw_text(ax, TIME_X, PARAM_VALUE_Y, time_str, LARGE_FONT_SIZE * 1.8, fontfamily="Noto Sans Math", height=LARGE_FONT_SIZE)
        else:
            draw_text(ax, TIME_X, PARAM_VALUE_Y, time_str, LARGE_FONT_SIZE)
            draw_text(ax, TIME_X, PARAM_UNIT_Y, time_unit_str, SMALL_FONT_SIZE)

        # 11 & 12: Temperature
        if self.set_temperature == 0:
            temp_str = "off"
        else:
            temp_str = f"{self.set_temperature:.0f} / {self.actual_temperature:.0f}"  # Show set / actual
        temp_unit_str = "°C"

        draw_text(ax, TEMP_X, PARAM_VALUE_Y, temp_str, LARGE_FONT_SIZE)
        draw_text(ax, TEMP_X, PARAM_UNIT_Y, temp_unit_str, SMALL_FONT_SIZE)

        # 10: Mixing Frequency
        if self.frequency == 0:
            freq_str = "0"
        else:
            freq_str = f"{self.frequency}"
        freq_unit_str = "rpm"

        draw_text(ax, FREQ_X, PARAM_VALUE_Y, freq_str, LARGE_FONT_SIZE)
        draw_text(ax, FREQ_X, PARAM_UNIT_Y, freq_unit_str, SMALL_FONT_SIZE)


@dataclass
class NoProgramMain:
    """Main mode without a program. The simplest mode."""

    status_bar: StatusBar
    main_parameter: MainParameter

    def draw(self, ax: plt.Axes):
        """Draws the simple main screen (status bar and parameters)."""
        ax.clear()
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)

        self.status_bar.draw(ax)
        self.main_parameter.draw(ax)
    
    @staticmethod
    def make_canvas():
        """Creates a new canvas for drawing."""
        fig = plt.figure(frameon=False, dpi=RESOLUTION)
        fig.set_size_inches(WIDTH / RESOLUTION, HEIGHT / RESOLUTION)
        ax = plt.Axes(fig, (0, 0, 1, 1), frameon=False)
        fig.add_axes(ax)
        return fig, ax

    @staticmethod
    def render_canvas(fig: plt.Figure):
        from io import BytesIO
        from PIL import Image
        io = BytesIO()
        fig.savefig(io, dpi=72 * 8, format="png")
        image = Image.open(io)
        image = image.convert("RGB")
        return np.array(image)
        
@dataclass
class Menu:
    """Menu mode. Entered when the user presses the 'Menu Enter' button."""
    # TODO

@dataclass
class ProgramMain:
    """
    Main mode with program. Entered when the user presses 'Prog 1~5' buttons,
    or when the program is selected in the menu.
    """
    # TODO

if __name__ == "__main__":
    fig = plt.figure(frameon=False, dpi=RESOLUTION)
    fig.set_size_inches(WIDTH / RESOLUTION, HEIGHT / RESOLUTION)
    ax = plt.Axes(fig, (0, 0, 1, 1), frameon=False)
    fig.add_axes(ax)

    status_bar_state = StatusBar(
        program_number=None,
        program_name=None,
        time_mode="temp_control",
        key_lock=False,
        speaker=True,
        interval_mix=False,
        thermotop=None,
        device_status=None,
    )
    main_param_state = MainParameter(
        time=Time(seconds=150),
        time_pause=False,
        set_temperature=45.0,
        actual_temperature=23.0,
        frequency=0,
    )
    ui_state = NoProgramMain(
        status_bar=status_bar_state, main_parameter=main_param_state
    )
    ui_state.draw(ax)
    plt.savefig("thermal_mixer_ui.png", dpi=72 * 8)

    def animate(i):
        if i % 20 == 0:
            ui_state.main_parameter.time.seconds -= 1
            ui_state.draw(ax)
        return fig

    ani = FuncAnimation(fig, animate, frames=10 * 20)
    # Use ffv1 codec for lossless compression and transparency support
    writer = FFMpegFileWriter(fps=20, codec='ffv1')
    ani.save("thermal_mixer_ui.mkv", writer=writer, dpi=72 * 8)


    ## Icon pixelation code
    # from PIL import Image
    # from pathlib import Path

    # def pixelate_icon(image_path, target_height: int):
    #     # Downscale screenshot of icon to pixelated size
    #     img = Image.open(image_path)

    #     target_width_float = target_height * img.width / img.height
    #     target_width = int(round(target_width_float))
    #     print(f"Target size: {target_width} ({target_width_float}) x {target_height}")
        
    #     # Binarize
    #     img = img.convert("L").point(lambda x: 0 if x < 50 else 255, '1')
    #     img = img.resize((target_width, target_height), Image.NEAREST)

    #     return img

    # for icon in Path("icons/raw").glob("*.png"):
    #     print(icon, end=": ")
    #     name, height = icon.stem.split("-")
    #     height = int(height)
    #     pixelate_icon(icon, height).save(f"icons/{name}.png")
