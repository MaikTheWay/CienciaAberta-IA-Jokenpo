"""
gui.py

Interface gráfica local para Pedra, Papel e Tesoura medieval.
Não altera a lógica original do jogo; apenas consome os módulos já existentes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk, ImageSequence

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from hand_detector import HandDetector
from game_logic import GameLogic, GameState  # noqa: E402


def _resample_nearest():
    return getattr(Image, "Resampling", Image).NEAREST


def _resample_lanczos():
    return getattr(Image, "Resampling", Image).LANCZOS


class GifSprite:
    """Carrega e anima GIFs com remoção simples do fundo escuro."""

    def __init__(self, path: Path, target_size: Tuple[int, int]):
        self.path = path
        self.target_size = target_size
        self.frames: list[ImageTk.PhotoImage] = []
        self.durations: list[int] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Asset não encontrado: {self.path}")

        gif = Image.open(self.path)
        frames = []
        durations = []

        for frame in ImageSequence.Iterator(gif):
            rgba = frame.convert("RGBA")
            arr = np.array(rgba)
            bg_color = arr[0, 0, :3].astype(np.int16)
            dist = np.sqrt(((arr[:, :, :3].astype(np.int16) - bg_color) ** 2).sum(axis=2))
            arr[:, :, 3] = np.where(dist <= 18, 0, 255).astype(np.uint8)
            processed = Image.fromarray(arr, "RGBA")
            processed = processed.resize(self.target_size, _resample_nearest())
            frames.append(ImageTk.PhotoImage(processed))
            durations.append(int(frame.info.get("duration", 120)))

        self.frames = frames
        self.durations = durations if durations else [120]

    def frame(self, index: int) -> ImageTk.PhotoImage:
        return self.frames[index % len(self.frames)]

    def duration(self, index: int) -> int:
        return self.durations[index % len(self.durations)]


class MedievalJokenpoGUI:
    BG_SIZE: Tuple[int, int] = (1672, 941)

    CAMERA_BOX = (1003, 155, 509, 334)  # x, y, w, h
    DIALOG_BOX = (520, 642, 703, 133)   # x, y, w, h
    WIZARD_POS = (100, 220)
    WIZARD_SIZE = (600, 600)
    SCORE_BOX = (620, 120, 435, 56)

    WAIT_PHRASES = [
        ("Bem-vindo, aventureiro!", "Mostre sua mão à câmera.", ""),
        ("A magia aguarda teu gesto.", "Posicione a mão diante do feitiço.", ""),
        ("Sinto uma presença no ar...", "Mantenha a mão visível para o ritual.", ""),
        ("O portal está aberto.", "Quando eu enxergar tua mão, iniciaremos.", ""),
    ]

    def __init__(self, detector: Optional[HandDetector] = None, game: Optional[GameLogic] = None):
        self.root = tk.Tk()
        self.root.title("Jokenpo Medieval")
        self.root.geometry(f"{self.BG_SIZE[0]}x{self.BG_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg="black")

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _e: self.close())
        self.root.bind("r", lambda _e: self.reset_round())
        self.root.bind("R", lambda _e: self.reset_round())

        self.bg_image = self._load_background()
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        self.canvas = tk.Canvas(
            self.root,
            width=self.BG_SIZE[0],
            height=self.BG_SIZE[1],
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=False)

        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        self.detector = detector or HandDetector()
        self.game = game or GameLogic(timer_seconds=3.0, final_window_seconds=0.2)

        self.cap = self._open_camera()
        self.camera_enabled = self.cap is not None and self.cap.isOpened()

        assets = BASE_DIR / "assets"
        self.sprite_idle = GifSprite(assets / "mago_parado.gif", self.WIZARD_SIZE)
        self.sprite_cast = GifSprite(assets / "mago_invocando.gif", self.WIZARD_SIZE)
        self.sprite_talk = GifSprite(assets / "mago_dialogando.gif", self.WIZARD_SIZE)

        self.current_sprite: Optional[GifSprite] = None
        self.sprite_index = 0
        self.sprite_after_id: Optional[str] = None
        self.wizard_item = self.canvas.create_image(
            self.WIZARD_POS[0],
            self.WIZARD_POS[1],
            anchor="nw",
        )

        cam_x, cam_y, cam_w, cam_h = self.CAMERA_BOX
        self.canvas.create_rectangle(
            cam_x,
            cam_y,
            cam_x + cam_w,
            cam_y + cam_h,
            outline="#ffffff",
            width=5,
            fill="#f7faf6",
        )
        self.camera_inner_margin = 7
        self.camera_target_size = (cam_w - self.camera_inner_margin * 2, cam_h - self.camera_inner_margin * 2)
        self.camera_item = self.canvas.create_image(
            cam_x + self.camera_inner_margin,
            cam_y + self.camera_inner_margin,
            anchor="nw",
        )
        self.camera_placeholder = self._create_placeholder_camera(self.camera_target_size)
        self.camera_photo = self.camera_placeholder

        dia_x, dia_y, dia_w, dia_h = self.DIALOG_BOX
        self.canvas.create_rectangle(
            dia_x,
            dia_y,
            dia_x + dia_w,
            dia_y + dia_h,
            outline="#ffffff",
            width=4,
            fill="#0b1a86",
        )

        score_x, score_y, score_w, score_h = self.SCORE_BOX
        self.canvas.create_rectangle(
            score_x,
            score_y,
            score_x + score_w,
            score_y + score_h,
            outline="#ffffff",
            width=3,
            fill="#1a1233",
        )
        self.score_title_item = self.canvas.create_text(
            score_x + score_w // 2,
            score_y + 14,
            anchor="center",
            text="PLACAR",
            fill="#f4f4f4",
            font=("Courier New", 11, "bold"),
        )
        self.score_item = self.canvas.create_text(
            score_x + score_w // 2,
            score_y + 36,
            anchor="center",
            text="MAGO 0  x  0 JOGADOR",
            fill="#ffffff",
            font=("Courier New", 16, "bold"),
        )

        self.dialog_main_font = ("Courier New", 16, "bold")
        self.dialog_sub_font = ("Courier New", 12, "bold")
        self.dialog_emphasis_font = ("Courier New", 28, "bold")

        self.dialog_main_item = self.canvas.create_text(
            dia_x + 20,
            dia_y + 14,
            anchor="nw",
            text="",
            fill="white",
            font=self.dialog_main_font,
            width=dia_w - 170,
        )
        self.dialog_sub_item = self.canvas.create_text(
            dia_x + 20,
            dia_y + 78,
            anchor="nw",
            text="",
            fill="#d6e9ff",
            font=self.dialog_sub_font,
            width=dia_w - 170,
        )
        self.dialog_emphasis_item = self.canvas.create_text(
            dia_x + dia_w - 90,
            dia_y + dia_h // 2,
            anchor="center",
            text="",
            fill="#fff3a0",
            font=self.dialog_emphasis_font,
        )

        self.running = True
        self.latest_snapshot = self.game.get_snapshot()

        self.dialog_state_signature = None
        self.dialog_visible_main = ""
        self.dialog_visible_sub = ""
        self.dialog_visible_emphasis = ""
        self.dialog_target_main = ""
        self.dialog_target_sub = ""
        self.dialog_target_emphasis = ""
        self.dialog_target_main_color = "white"
        self.dialog_target_sub_color = "#d6e9ff"
        self.dialog_target_emphasis_color = "#fff3a0"
        self.dialog_chars_per_tick = 2
        self.dialog_after_id: Optional[str] = None
        self.dialog_tick_delay = 28
        self.dialog_last_tick = 0.0
        self.wait_phrase_index = 0
        self.wait_phrase_last_switch = 0.0
        self.wait_phrase_interval = 2.8
        self.current_dialog_base = None

        self._set_sprite(self.sprite_idle)
        self._loop()

    def _load_background(self) -> Image.Image:
        candidates = [
            BASE_DIR / "assets" / "cenario.png",
            BASE_DIR / "assets" / "Cenario.png",
            BASE_DIR / "assets" / "Cenário.png",
        ]
        for p in candidates:
            if p.exists():
                bg = Image.open(p).convert("RGBA")
                if bg.size != self.BG_SIZE:
                    bg = bg.resize(self.BG_SIZE, _resample_nearest())
                return bg
        raise FileNotFoundError("Não foi possível localizar o cenário em assets/.")

    def _open_camera(self):
        for index in (1, 0):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    def _create_placeholder_camera(self, size: Tuple[int, int]) -> ImageTk.PhotoImage:
        w, h = size
        img = Image.new("RGB", size, (245, 248, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, w - 1, h - 1), outline=(255, 255, 255), width=3)
        text = "CÂMERA\nINDISPONÍVEL"
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=8, align="center")
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (w - tw) // 2
        y = (h - th) // 2
        draw.multiline_text((x, y), text, fill=(70, 95, 70), font=font, spacing=8, align="center")
        return ImageTk.PhotoImage(img)

    def _set_sprite(self, sprite: GifSprite) -> None:
        if self.current_sprite is sprite:
            return
        self.current_sprite = sprite
        self.sprite_index = 0
        if self.sprite_after_id is not None:
            try:
                self.root.after_cancel(self.sprite_after_id)
            except Exception:
                pass
            self.sprite_after_id = None
        self._update_sprite_frame()

    def _update_sprite_frame(self) -> None:
        if not self.running or self.current_sprite is None:
            return

        frame = self.current_sprite.frame(self.sprite_index)
        self.canvas.itemconfig(self.wizard_item, image=frame)
        self.wizard_photo = frame  # mantém a referência do frame atual

        delay = max(40, self.current_sprite.duration(self.sprite_index))
        self.sprite_index = (self.sprite_index + 1) % len(self.current_sprite.frames)
        self.sprite_after_id = self.root.after(delay, self._update_sprite_frame)

    def _frame_to_photo(self, frame_bgr: np.ndarray) -> ImageTk.PhotoImage:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        pil = ImageOps.fit(pil, self.camera_target_size, method=_resample_lanczos())
        return ImageTk.PhotoImage(pil)

    def _select_wait_phrase(self, hand_visible: bool) -> Tuple[str, str, str]:
        now = time.time()
        if self.wait_phrase_last_switch == 0.0:
            self.wait_phrase_last_switch = now

        if now - self.wait_phrase_last_switch >= self.wait_phrase_interval:
            self.wait_phrase_index = (self.wait_phrase_index + 1) % len(self.WAIT_PHRASES)
            self.wait_phrase_last_switch = now

        main_text, sub_text, emph = self.WAIT_PHRASES[self.wait_phrase_index]
        if hand_visible:
            main_text = "Perfeito... eu sinto a tua presença."
            sub_text = "Mantenha a mão firme, aventureiro."
        return main_text, sub_text, emph

    def _dialogue_content(self, snapshot, hand_visible: bool) -> Tuple[str, str, str, str, str, str]:
        state = snapshot["state"]
        timer_text = snapshot["timer_text"]
        result = snapshot["result"]

        if state == GameState.WAIT_HAND:
            main_text, sub_text, emph = self._select_wait_phrase(hand_visible)
            return (
                main_text,
                sub_text,
                emph,
                "white",
                "#d6e9ff",
                "#fff3a0",
            )

        if state == GameState.COUNTDOWN:
            return (
                "Prepare-se! O desafio está começando...",
                "Concentre-se no feitiço.",
                f"{timer_text}",
                "#fff2a8",
                "#f2ffb0",
                "#fff3a0",
            )

        if "JOGADOR VENCEU" in result:
            return (
                "Vitória! Teu golpe foi digno de um cavaleiro.",
                "O reino celebra a tua coragem.",
                "VITÓRIA",
                "#8cff94",
                "#8cff94",
                "#8cff94",
            )

        if "IA VENCEU" in result:
            return (
                "A coroa permanece comigo. Tente novamente, bravo aventureiro.",
                "A magia foi mais forte desta vez.",
                "DERROTA",
                "#ff9090",
                "#ff9090",
                "#ff9090",
            )

        if "EMPATE" in result:
            return (
                "Empate! Duas lâminas, nenhum vencedor.",
                "Reinicie o duelo e tente outra vez.",
                "EMPATE",
                "#ffd96a",
                "#ffd96a",
                "#ffd96a",
            )

        return (
            "A magia se dissipou... Tente novamente.",
            result,
            "SEM RESULTADO",
            "#e8e8e8",
            "#e8e8e8",
            "#e8e8e8",
        )

    def _set_dialog_target(self, main_text: str, sub_text: str, emph_text: str, main_color: str, sub_color: str, emph_color: str) -> None:
        target = (main_text, sub_text, emph_text, main_color, sub_color, emph_color)
        if target == self.current_dialog_base:
            return
        self.current_dialog_base = target
        self.dialog_target_main = main_text
        self.dialog_target_sub = sub_text
        self.dialog_target_emphasis = emph_text
        self.dialog_target_main_color = main_color
        self.dialog_target_sub_color = sub_color
        self.dialog_target_emphasis_color = emph_color
        self.dialog_visible_main = ""
        self.dialog_visible_sub = ""
        self.dialog_visible_emphasis = ""
        self.dialog_last_tick = 0.0
        self._refresh_dialog_text(reset_colors=True)

    def _refresh_dialog_text(self, reset_colors: bool = False) -> None:
        self.canvas.itemconfig(self.dialog_main_item, text=self.dialog_visible_main, fill=self.dialog_target_main_color)
        self.canvas.itemconfig(self.dialog_sub_item, text=self.dialog_visible_sub, fill=self.dialog_target_sub_color)
        self.canvas.itemconfig(self.dialog_emphasis_item, text=self.dialog_visible_emphasis, fill=self.dialog_target_emphasis_color)
        if reset_colors:
            self.canvas.itemconfig(self.dialog_main_item, fill=self.dialog_target_main_color)
            self.canvas.itemconfig(self.dialog_sub_item, fill=self.dialog_target_sub_color)
            self.canvas.itemconfig(self.dialog_emphasis_item, fill=self.dialog_target_emphasis_color)

    def _advance_text(self, current: str, target: str) -> str:
        if current == target:
            return current
        next_len = min(len(target), len(current) + self.dialog_chars_per_tick)
        return target[:next_len]

    def _animate_dialogue(self) -> None:
        if not self.running:
            return

        now = time.time()
        if now - self.dialog_last_tick < (self.dialog_tick_delay / 1000.0):
            self.dialog_after_id = self.root.after(20, self._animate_dialogue)
            return
        self.dialog_last_tick = now

        self.dialog_visible_main = self._advance_text(self.dialog_visible_main, self.dialog_target_main)
        self.dialog_visible_sub = self._advance_text(self.dialog_visible_sub, self.dialog_target_sub)
        self.dialog_visible_emphasis = self._advance_text(self.dialog_visible_emphasis, self.dialog_target_emphasis)
        self._refresh_dialog_text()

        if (
            self.dialog_visible_main != self.dialog_target_main
            or self.dialog_visible_sub != self.dialog_target_sub
            or self.dialog_visible_emphasis != self.dialog_target_emphasis
        ):
            self.dialog_after_id = self.root.after(20, self._animate_dialogue)
        else:
            self.dialog_after_id = None

    def _update_camera(self, frame_bgr: Optional[np.ndarray]) -> None:
        if frame_bgr is None:
            photo = self.camera_placeholder
        else:
            photo = self._frame_to_photo(frame_bgr)
        self.canvas.itemconfig(self.camera_item, image=photo)
        self.camera_photo = photo

    def _set_wizard_state(self, state: GameState) -> None:
        if state == GameState.WAIT_HAND:
            self._set_sprite(self.sprite_idle)
        elif state == GameState.COUNTDOWN:
            self._set_sprite(self.sprite_cast)
        else:
            self._set_sprite(self.sprite_talk)

    def _update_scoreboard(self) -> None:
        stats = self.game.get_statistics()
        score_text = f"MAGO {stats.get('ai_wins', 0)}  x  {stats.get('player_wins', 0)} JOGADOR"
        self.canvas.itemconfig(self.score_item, text=score_text)

    def _loop(self) -> None:
        if not self.running:
            return

        frame_bgr = None
        hand_visible = False
        landmarks = []
        handedness = None

        if self.camera_enabled and self.cap is not None:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                frame = cv2.flip(frame, 1)
                detection = self.detector.encontrar_pontos(frame)
                frame_bgr = detection.annotated_frame
                hand_visible = detection.visible
                landmarks = detection.landmarks
                handedness = detection.handedness

        self.latest_snapshot = self.game.update(
            hand_visible=hand_visible,
            landmarks=landmarks,
            handedness=handedness,
        )

        self._set_wizard_state(self.latest_snapshot["state"])
        main_text, sub_text, emph_text, main_color, sub_color, emph_color = self._dialogue_content(self.latest_snapshot, hand_visible)
        self._set_dialog_target(main_text, sub_text, emph_text, main_color, sub_color, emph_color)
        if self.dialog_after_id is None:
            self._animate_dialogue()

        self._update_camera(frame_bgr)
        self._update_scoreboard()

        self.root.after(15, self._loop)

    def reset_round(self) -> None:
        self.game.reset_round()
        self.wait_phrase_index = 0
        self.wait_phrase_last_switch = 0.0
        self.current_dialog_base = None
        self.dialog_visible_main = ""
        self.dialog_visible_sub = ""
        self.dialog_visible_emphasis = ""
        self.dialog_target_main = ""
        self.dialog_target_sub = ""
        self.dialog_target_emphasis = ""
        self.dialog_last_tick = 0.0
        self._refresh_dialog_text(reset_colors=True)

    def close(self) -> None:
        if not self.running:
            return
        self.running = False

        if self.sprite_after_id is not None:
            try:
                self.root.after_cancel(self.sprite_after_id)
            except Exception:
                pass
            self.sprite_after_id = None

        if self.dialog_after_id is not None:
            try:
                self.root.after_cancel(self.dialog_after_id)
            except Exception:
                pass
            self.dialog_after_id = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()


def main():
    app = MedievalJokenpoGUI()
    app.run()


if __name__ == "__main__":
    main()
