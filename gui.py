"""
gui.py

Interface gráfica local para Pedra, Papel e Tesoura medieval.
Não altera a lógica original do jogo; apenas consome os módulos já existentes.
"""

from __future__ import annotations

import random
import sys
import time
import colorsys
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    DIALOG_BOX = (470, 610, 770, 220)
    WIZARD_POS = (100, 220)             # x, y
    WIZARD_SIZE = (600, 600)
    SCORE_BOX = (540, 28, 590, 96)

    WAIT_PHRASES = [
        ("Bem-vindo, aventureiro!", "Mostre sua mão à câmera.", ""),
        ("Você parece confiante...", "Isso costuma acabar mal.", ""),
        ("Sinto uma presença no ar...", "Mantenha a mão visível para o ritual.", ""),
        ("Estou canalizando energias antigas...", "E também esperando sua mão aparecer.", ""),
        ("Ah, outro herói corajoso!", "Ou apenas alguém confiante demais.", ""),
        ("Não me faça te transformar em um golbin.", "Posicione sua mão corretamente!", ""),
        ("Eu não enxergo mãos escondidas.", "Exponha teu gesto ao destino.", ""),
        ("Interessante...", "Você realmente acha que pode vencer?", ""),
        ("Vejo coragem em seus olhos...", "Ou é só falta de noção mesmo?", "")
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
        
        # Atalhos para mudança de modo de jogo (Silencioso)
        self.root.bind("1", lambda _e: self.game.set_game_mode(1))
        self.root.bind("2", lambda _e: self.game.set_game_mode(2))
        self.root.bind("3", lambda _e: self.game.set_game_mode(3))

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
        self.spell_sprite = GifSprite(assets / "feitico.gif", (130, 130))
        self.move_assets = self._load_move_assets(assets)
        self.font_path = self._resolve_font_path()
        self._image_refs: Dict[str, ImageTk.PhotoImage] = {}

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
        self.dialog_box_item = self.canvas.create_rectangle(
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
        # Ajuste fino da centralização do placar
        score_center_x = score_x + score_w // 2
        self.score_title_item = self.canvas.create_image(score_center_x, score_y + 24, anchor="center")
        self.score_item = self.canvas.create_image(score_center_x, score_y + 60, anchor="center")

        # Layout ajustado para centralizar elementos
        self.dialog_main_item = self.canvas.create_image(dia_x + dia_w // 2, dia_y + 70, anchor="center")
        self.dialog_sub_item = self.canvas.create_image(dia_x + dia_w // 2, dia_y + 150, anchor="center")
        self.dialog_emphasis_item = self.canvas.create_image(dia_x + dia_w // 2, dia_y + dia_h // 2, anchor="center")
        self.dialog_move_item = self.canvas.create_image(dia_x + dia_w // 2, dia_y + dia_h // 2, anchor="center")
        self.spell_item = self.canvas.create_image(dia_x + dia_w // 2, dia_y + dia_h // 2, anchor="center")
        self.dialog_move_photo = None
        self.spell_photo = None

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
        self.dialog_target_asset_key: Optional[str] = None
        
        self.typewriter_after_id: Optional[str] = None
        self.typewriter_index_main = 0
        self.typewriter_index_sub = 0
        
        self.dialog_visible_until = 0.0
        self.dialog_hidden_until = 0.0
        self.dialog_showing = False
        self.wait_phrase_last_index = -1

        self.result_phase = "idle"  # idle -> freeze -> dialog
        self.result_freeze_until = 0.0
        self.result_dialog_until = 0.0
        self.result_message = ""
        self.result_message_kind = "draw"
        self.result_move_key: Optional[str] = None

        self._set_sprite(self.sprite_idle)
        self._loop()

    def _load_background(self) -> Image.Image:
        candidates = [
            BASE_DIR / "assets" / "cenario.png"
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

    def _resolve_font_path(self) -> Path:
        candidates = [BASE_DIR / "assets" / "cornelia-sans.otf", BASE_DIR / "cornelia-sans.otf"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path("")

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        if self.font_path and self.font_path.exists():
            try:
                return ImageFont.truetype(str(self.font_path), size)
            except Exception:
                pass
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def _fit_text_lines(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
        if not text:
            return ""
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        lines = []
        for paragraph in text.split("\n"):
            words = paragraph.split()
            if not words:
                lines.append("")
                continue
            current_line = []
            for word in words:
                test_line = " ".join(current_line + [word])
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] <= max_width:
                    current_line.append(word)
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
            lines.append(" ".join(current_line))
        return "\n".join(lines)

    def _load_move_assets(self, assets_dir: Path) -> Dict[str, ImageTk.PhotoImage]:
        mapping = {
            "PEDRA": "mao_pedra.png",
            "PAPEL": "mao_papel.png",
            "TESOURA": "mao_tesoura.png",
        }
        loaded = {}
        for key, filename in mapping.items():
            path = assets_dir / filename
            if path.exists():
                img = Image.open(path).convert("RGBA")
                # Aumentar tamanho da mão para centralizar melhor
                img = img.resize((165, 165), _resample_lanczos())
                loaded[key] = ImageTk.PhotoImage(img)
        return loaded

    def _render_text_photo(
        self,
        text: str,
        font_size: int,
        fill: str = "white",
        max_width: int = 600,
        max_height: int = 150,
        align: str = "left",
        line_spacing: int = 4,
        padding: Tuple[int, int] = (10, 10),
        min_size: Optional[Tuple[int, int]] = None,
    ) -> ImageTk.PhotoImage:
        if not text or text.strip() == "":
            text = " "
        font = self._load_font(font_size)
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        
        # Obter o bounding box exato do texto
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=line_spacing, align=align)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Fontes FreeType podem ter descendentes que ultrapassam o bbox
        # Adicionamos uma margem de segurança baseada no tamanho da fonte
        safety_margin = font_size // 3
        
        canvas_w = max(tw + padding[0] * 2 + safety_margin, min_size[0] if min_size else 0)
        canvas_h = max(th + padding[1] * 2 + safety_margin, min_size[1] if min_size else 0)
        
        canvas_w = min(canvas_w, max_width)
        canvas_h = min(canvas_h, max_height)

        img = Image.new("RGBA", (int(canvas_w), int(canvas_h)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Centralização precisa considerando o deslocamento do bbox (âncora)
        tx = (canvas_w - tw) / 2 - bbox[0]
        ty = (canvas_h - th) / 2 - bbox[1]
        
        draw.multiline_text((tx, ty), text, font=font, fill=fill, spacing=line_spacing, align=align)
        return ImageTk.PhotoImage(img)

    def _set_image_item(self, item_id: int, photo: ImageTk.PhotoImage, ref_key: str) -> None:
        self._image_refs[ref_key] = photo
        self.canvas.itemconfig(item_id, image=photo)

    def _render_special_text_photo(
        self,
        text: str,
        font_size: int,
        mode: str,
        max_width: int = 720,
        max_height: int = 180,
        line_spacing: int = 4,
        padding: Tuple[int, int] = (4, 4),
    ) -> ImageTk.PhotoImage:
        # Texto especial do resultado: RGB, piscando ou laranja.
        if not text:
            return self._render_text_photo(
                '',
                font_size=font_size,
                fill='white',
                max_width=max_width,
                max_height=max_height,
                align='center',
                line_spacing=line_spacing,
                padding=padding,
                min_size=(120, 40),
            )

        font = self._load_font(font_size)
        helper = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(helper)
        wrapped = self._fit_text_lines(text, font, max_width - 10)
        lines = wrapped.splitlines() or [wrapped]

        line_metrics = []
        total_h = 0
        max_line_w = 0
        for line in lines:
            if not line:
                bbox = draw.textbbox((0, 0), 'A', font=font)
                lh = bbox[3] - bbox[1]
                line_metrics.append((0, lh, []))
                total_h += lh + line_spacing
                continue
            chars = []
            line_w = 0
            line_h = 0
            for ch in line:
                bbox = draw.textbbox((0, 0), ch, font=font)
                cw = max(1, bbox[2] - bbox[0])
                ch_h = max(1, bbox[3] - bbox[1])
                chars.append((ch, cw, ch_h))
                line_w += cw
                line_h = max(line_h, ch_h)
            line_metrics.append((line_w, line_h, chars))
            max_line_w = max(max_line_w, line_w)
            total_h += line_h + line_spacing

        total_h = max(1, total_h - line_spacing)

        # Adicionamos uma margem de segurança extra para evitar cortes
        safety_margin = font_size // 3
        canvas_w = max(120, min(max_width, max_line_w + padding[0] * 2 + safety_margin))
        canvas_h = max(40, min(max_height, total_h + padding[1] * 2 + safety_margin))

        img = Image.new('RGBA', (int(canvas_w), int(canvas_h)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        now = time.time()
        
        # Centralização vertical
        y = (canvas_h - total_h) / 2
        hue_shift = (now * 120.0) % 360.0

        for line_w, line_h, chars in line_metrics:
            # Centralização horizontal por linha
            x = (canvas_w - line_w) / 2
            for idx, (ch, cw, ch_h) in enumerate(chars):
                if mode == 'rainbow':
                    hue = ((idx * 24.0) + hue_shift) % 360.0 / 360.0
                    rgb = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
                    fill = tuple(int(v * 255) for v in rgb) + (255,)
                elif mode == 'blink_red':
                    fill = (255, 40, 40, 255) if int(now * 4) % 2 == 0 else (120, 0, 0, 255)
                else:
                    fill = (255, 152, 0, 255)
                
                # Usamos anchor='la' (left, ascender) para melhor controle de alinhamento
                # O y é ajustado para centralizar com base na altura da linha
                char_y = y + (line_h - ch_h) / 2
                draw.text((x, char_y), ch, font=font, fill=fill)
                x += cw
            y += line_h + line_spacing

        return ImageTk.PhotoImage(img)

    def _random_dialog_duration(self) -> float:
        return random.uniform(4.0, 6.5)

    def _random_hidden_gap(self) -> float:
        return random.uniform(1.0, 2.5)

    def _pick_wait_phrase(self) -> Tuple[str, str, str]:
        if not self.WAIT_PHRASES:
            return ("...", "", "")
        if len(self.WAIT_PHRASES) == 1:
            return self.WAIT_PHRASES[0]
        index = random.randrange(len(self.WAIT_PHRASES))
        if index == self.wait_phrase_last_index:
            index = (index + 1) % len(self.WAIT_PHRASES)
        self.wait_phrase_last_index = index
        return self.WAIT_PHRASES[index]

    def _show_dialog(self, main_text: str, sub_text: str, emph_text: str, main_color: str, sub_color: str, emph_color: str, asset_key: Optional[str] = None) -> None:
        # Fontes aumentadas
        font_main = self._load_font(30)
        font_sub = self._load_font(24)
        max_w = self.DIALOG_BOX[2] - 90 # mais espaço visual e menos risco de corte
        
        self.dialog_target_main = self._fit_text_lines(main_text, font_main, max_w)
        self.dialog_target_sub = self._fit_text_lines(sub_text, font_sub, max_w)
        self.dialog_target_emphasis = emph_text
        self.dialog_target_main_color = main_color
        self.dialog_target_sub_color = sub_color
        self.dialog_target_emphasis_color = emph_color
        self.dialog_target_asset_key = asset_key
        
        # Resetar animação de escrita
        self.dialog_visible_main = ""
        self.dialog_visible_sub = ""
        self.dialog_visible_emphasis = emph_text 
        self.typewriter_index_main = 0
        self.typewriter_index_sub = 0
        
        self.dialog_visible_until = time.time() + self._random_dialog_duration()
        self.dialog_showing = True
        
        if self.typewriter_after_id:
            self.root.after_cancel(self.typewriter_after_id)
        
        # Contagem regressiva fica limpa e centralizada
        if emph_text in ["3", "2", "1", "JÁ!"]:
            self.dialog_visible_main = ""
            self.dialog_visible_sub = ""
            self.dialog_visible_emphasis = emph_text
            self._refresh_dialog_text()
        else:
            self._typewriter_step()
        
        self._update_dialog_asset()
        self._refresh_dialog_visibility(True)

    def _typewriter_step(self) -> None:
        if not self.dialog_showing:
            return

        changed = False
        if self.typewriter_index_main < len(self.dialog_target_main):
            self.typewriter_index_main += 1
            self.dialog_visible_main = self.dialog_target_main[:self.typewriter_index_main]
            changed = True
        elif self.typewriter_index_sub < len(self.dialog_target_sub):
            self.typewriter_index_sub += 1
            self.dialog_visible_sub = self.dialog_target_sub[:self.typewriter_index_sub]
            changed = True
        
        if changed:
            self._refresh_dialog_text()
            self.typewriter_after_id = self.root.after(20, self._typewriter_step)
        else:
            self.typewriter_after_id = None

    def _hide_dialog(self) -> None:
        self.dialog_showing = False
        self.dialog_hidden_until = time.time() + self._random_hidden_gap()
        self._refresh_dialog_visibility(False)

    def _refresh_dialog_visibility(self, visible: bool) -> None:
        state = "normal" if visible else "hidden"
        self.canvas.itemconfig(self.dialog_box_item, state=state)
        self.canvas.itemconfig(self.dialog_main_item, state=state)
        self.canvas.itemconfig(self.dialog_sub_item, state=state)
        self.canvas.itemconfig(self.dialog_emphasis_item, state=state)
        self.canvas.itemconfig(self.dialog_move_item, state=state)

    def _update_dialog_asset(self) -> None:
        asset = self.move_assets.get(self.dialog_target_asset_key or "")
        self.dialog_move_photo = asset
        self.canvas.itemconfig(self.dialog_move_item, image=asset if asset is not None else "")

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

        # Durante o congelamento do resultado, trava a animação do mago.
        if self.result_phase == "freeze":
            delay = 60
            self.sprite_after_id = self.root.after(delay, self._update_sprite_frame)
            return

        frame = self.current_sprite.frame(self.sprite_index)
        self.canvas.itemconfig(self.wizard_item, image=frame)
        self.wizard_photo = frame 

        delay = max(40, self.current_sprite.duration(self.sprite_index))
        self.sprite_index = (self.sprite_index + 1) % len(self.current_sprite.frames)
        self.sprite_after_id = self.root.after(delay, self._update_sprite_frame)

    def _frame_to_photo(self, frame_bgr: np.ndarray) -> ImageTk.PhotoImage:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        pil = ImageOps.fit(pil, self.camera_target_size, method=_resample_lanczos())
        return ImageTk.PhotoImage(pil)

    def _select_wait_phrase(self, hand_visible: bool) -> Tuple[str, str, str]:
        main_text, sub_text, emph = self._pick_wait_phrase()
        if hand_visible:
            main_text = "Perfeito... eu sinto a tua presença."
            sub_text = "Mantenha a mão firme, aventureiro."
        return main_text, sub_text, emph

    def _dialogue_content(self, snapshot, hand_visible: bool) -> Tuple[str, str, str, str, str, str, Optional[str]]:
        state = snapshot["state"]
        timer_text = snapshot["timer_text"]
        result = snapshot["result"]
        ai_move = snapshot.get("ai_move") or ""

        if state == GameState.WAIT_HAND:
            main_text, sub_text, emph = self._select_wait_phrase(hand_visible)
            return (main_text, sub_text, emph, "white", "#d6e9ff", "#fff3a0", None)

        if state == GameState.COUNTDOWN:
            return ("", "", f"{timer_text}", "#ff2b2b", "#ff2b2b", "#ff2b2b", None)

        if "JOGADOR VENCEU" in result:
            emph = "Você vençeu! Meus parabéns!"
            color = "#8cff94"
        elif "IA VENCEU" in result:
            emph = "VOCÊ PERDEU! HAHAHA"
            color = "#ff4b4b"
        else:
            emph = "EMPATAMOS! HAHAHA"
            color = "#ff9800"

        asset_key = ai_move if ai_move in self.move_assets else None
        return ("", "", emph, color, color, color, asset_key)

    def _refresh_dialog_text(self, reset_colors: bool = False) -> None:
        stats = self.game.get_statistics()

        main_text = self.dialog_visible_main
        sub_text = self.dialog_visible_sub
        emph_text = self.dialog_visible_emphasis

        # Renderizamos com base no texto visível atual da animação
        main_photo = self._render_text_photo(
            main_text,
            font_size=30,
            fill=self.dialog_target_main_color,
            max_width=self.DIALOG_BOX[2] - 40,
            max_height=100,
            align="center",
            line_spacing=5,
            padding=(10, 10),
            min_size=(self.DIALOG_BOX[2] - 60, 40),
        )
        sub_photo = self._render_text_photo(
            sub_text,
            font_size=24,
            fill=self.dialog_target_sub_color,
            max_width=self.DIALOG_BOX[2] - 40,
            max_height=80,
            align="center",
            line_spacing=4,
            padding=(10, 10),
            min_size=(self.DIALOG_BOX[2] - 60, 32),
        )

        # Fonte do timer/ênfase muito maior quando centralizado
        if emph_text in ["3", "2", "1", "JÁ!"]:
            emph_photo = self._render_text_photo(
                emph_text,
                font_size=160,  # Aumentado para preencher melhor o box
                fill="#ff1e1e",
                max_width=self.DIALOG_BOX[2],
                max_height=self.DIALOG_BOX[3],
                align="center",
                padding=(0, 0),
                min_size=(self.DIALOG_BOX[2], self.DIALOG_BOX[3]),
            )
        elif self.result_phase == "dialog" and emph_text:
            if self.result_message_kind == "win":
                emph_photo = self._render_special_text_photo(
                    emph_text,
                    font_size=48,
                    mode="rainbow",
                    max_width=self.DIALOG_BOX[2] - 50,
                    max_height=180,
                    line_spacing=6,
                    padding=(8, 8),
                )
            elif self.result_message_kind == "loss":
                emph_photo = self._render_special_text_photo(
                    emph_text,
                    font_size=52,
                    mode="blink_red",
                    max_width=self.DIALOG_BOX[2] - 50,
                    max_height=180,
                    line_spacing=6,
                    padding=(8, 8),
                )
            else:
                emph_photo = self._render_special_text_photo(
                    emph_text,
                    font_size=50,
                    mode="orange",
                    max_width=self.DIALOG_BOX[2] - 50,
                    max_height=180,
                    line_spacing=6,
                    padding=(8, 8),
                )
        else:
            emph_photo = self._render_text_photo(
                emph_text,
                font_size=36 if emph_text else 20,
                fill=self.dialog_target_emphasis_color,
                max_width=self.DIALOG_BOX[2] - 50,
                max_height=170,
                align="center",
                padding=(4, 2),
                min_size=(150, 50),
            )

        score_title_photo = self._render_text_photo(
            "PLACAR",
            font_size=24,
            fill="#f4f4f4",
            align="center",
            padding=(2, 2),
            min_size=(160, 28),
        )
        score_text = f"VOCÊ: {stats['player_wins']}  |  MAGO: {stats['ai_wins']}  |  EMPATES: {stats['draws']}"
        score_photo = self._render_text_photo(
            score_text,
            font_size=24,
            fill="#ffffff",
            align="center",
            padding=(2, 2),
            min_size=(self.SCORE_BOX[2] - 20, 34),
        )

        self._set_image_item(self.dialog_main_item, main_photo, "dialog_main")
        self._set_image_item(self.dialog_sub_item, sub_photo, "dialog_sub")
        self._set_image_item(self.dialog_emphasis_item, emph_photo, "dialog_emphasis")
        self._set_image_item(self.score_title_item, score_title_photo, "score_title")
        self._set_image_item(self.score_item, score_photo, "score_value")

        if reset_colors:
            self.canvas.itemconfig(self.dialog_main_item, state="normal")
            self.canvas.itemconfig(self.dialog_sub_item, state="normal")
            self.canvas.itemconfig(self.dialog_emphasis_item, state="normal")

    def _set_dialog_target(self, main_text: str, sub_text: str, emph_text: str, main_color: str, sub_color: str, emph_color: str, asset_key: Optional[str] = None) -> None:
        target = (main_text, sub_text, emph_text, main_color, sub_color, emph_color, asset_key)
        if target == self.dialog_state_signature:
            return
        self.dialog_state_signature = target
        self._show_dialog(main_text, sub_text, emph_text, main_color, sub_color, emph_color, asset_key)

    def _begin_result_sequence(self, snapshot) -> None:
        # Congela a tela por 3s e depois mostra o texto final.
        now = time.time()
        self.result_phase = "freeze"
        self.result_freeze_until = now + 3.0
        self.result_dialog_until = 0.0

        result = snapshot.get("result") or ""
        if "JOGADOR VENCEU" in result:
            self.result_message_kind = "win"
            self.result_message = "Você vençeu! Meus parabéns!"
        elif "IA VENCEU" in result:
            self.result_message_kind = "loss"
            self.result_message = "VOCÊ PERDEU! HAHAHA"
        else:
            self.result_message_kind = "draw"
            self.result_message = "EMPATAMOS! HAHAHA"

        self.result_move_key = snapshot.get("ai_move") if snapshot.get("ai_move") in self.move_assets else None
        self.dialog_visible_main = ""
        self.dialog_visible_sub = ""
        self.dialog_visible_emphasis = ""
        self.dialog_target_main = ""
        self.dialog_target_sub = ""
        self.dialog_target_emphasis = ""
        self.dialog_target_asset_key = self.result_move_key
        self._refresh_result_overlay(visible=True, show_text=False)

    def _refresh_result_overlay(self, visible: bool, show_text: bool) -> None:
        dia_x, dia_y, dia_w, dia_h = self.DIALOG_BOX
        state = "normal" if visible else "hidden"

        self.canvas.itemconfig(self.dialog_box_item, state=state)
        self.canvas.itemconfig(self.dialog_main_item, state="hidden")
        self.canvas.itemconfig(self.dialog_sub_item, state="hidden")
        self.canvas.itemconfig(self.dialog_emphasis_item, state=state if show_text else "hidden")
        self.canvas.itemconfig(self.dialog_move_item, state=state if visible and not show_text else "hidden")
        self.canvas.itemconfig(self.spell_item, state=state if visible and not show_text else "hidden")

        if visible and not show_text:
            move_photo = self.move_assets.get(self.result_move_key or "")
            self.dialog_move_photo = move_photo
            self.canvas.itemconfig(self.dialog_move_item, image=move_photo if move_photo is not None else "")

            spell_photo = self.spell_sprite.frame(0)
            self.spell_photo = spell_photo
            self.canvas.itemconfig(self.spell_item, image=spell_photo)

            move_center_x = dia_x + dia_w // 2
            move_center_y = dia_y + dia_h // 2
            self.canvas.coords(self.dialog_move_item, move_center_x, move_center_y)
            self.canvas.coords(self.spell_item, move_center_x, move_center_y)

    def _update_dialog_cycle(self, snapshot, hand_visible: bool) -> None:
        state = snapshot["state"]
        now = time.time()

        if state == GameState.WAIT_HAND:
            self.result_phase = "idle"
            self.result_move_key = None

            if self.dialog_showing:
                if now >= self.dialog_visible_until:
                    self._hide_dialog()
                return

            if now < self.dialog_hidden_until:
                return

            main_text, sub_text, emph_text, main_color, sub_color, emph_color, asset_key = self._dialogue_content(snapshot, hand_visible)
            self._set_dialog_target(main_text, sub_text, emph_text, main_color, sub_color, emph_color, asset_key)
            return

        if state == GameState.COUNTDOWN:
            main_text, sub_text, emph_text, main_color, sub_color, emph_color, asset_key = self._dialogue_content(snapshot, hand_visible)
            self._set_dialog_target(main_text, sub_text, emph_text, main_color, sub_color, emph_color, asset_key)
            self.dialog_visible_until = now + 0.2
            self.dialog_hidden_until = 0.0
            return

        # RESULTADO
        if self.result_phase == "idle":
            self._begin_result_sequence(snapshot)
            return

        if self.result_phase == "freeze":
            self._refresh_result_overlay(visible=True, show_text=False)
            if now >= self.result_freeze_until:
                self.result_phase = "dialog"
                self.result_dialog_until = now + 3.2
                self._set_sprite(self.sprite_talk)
                self.dialog_visible_emphasis = self.result_message
                self._refresh_result_overlay(visible=True, show_text=True)
                self._refresh_dialog_text()
            return

        if self.result_phase == "dialog":
            self._refresh_result_overlay(visible=True, show_text=True)
            self.dialog_visible_emphasis = self.result_message
            self._refresh_dialog_text()
            if now >= self.result_dialog_until:
                self.reset_round()
            return

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
        if not self.typewriter_after_id:
            self._refresh_dialog_text(reset_colors=False)

    def _loop(self) -> None:
        if not self.running:
            return

        frame_bgr = None
        hand_visible = False
        landmarks = []
        handedness = None

        result_active = self.result_phase in {"freeze", "dialog"}

        if self.camera_enabled and self.cap is not None and not result_active:
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

        # Se a rodada acabou, bloqueia a câmera e alterna os sprites em fases locais.
        if self.latest_snapshot["state"] == GameState.WAIT_HAND:
            self._set_wizard_state(GameState.WAIT_HAND)
        elif self.latest_snapshot["state"] == GameState.COUNTDOWN:
            self._set_wizard_state(GameState.COUNTDOWN)
        else:
            if self.result_phase == "freeze":
                self._set_sprite(self.sprite_cast)
            else:
                self._set_sprite(self.sprite_talk)

        self._update_dialog_cycle(self.latest_snapshot, hand_visible)

        if not result_active:
            self._update_camera(frame_bgr)
        self._update_scoreboard()

        self.root.after(15, self._loop)

    def reset_round(self) -> None:
        self.game.reset_round()
        self.dialog_state_signature = None
        self.dialog_visible_main = ""
        self.dialog_visible_sub = ""
        self.dialog_visible_emphasis = ""
        self.dialog_target_main = ""
        self.dialog_target_sub = ""
        self.dialog_target_emphasis = ""
        self.dialog_target_asset_key = None
        self.dialog_visible_until = 0.0
        self.dialog_hidden_until = 0.0
        self.dialog_showing = False

        self.result_phase = "idle"
        self.result_freeze_until = 0.0
        self.result_dialog_until = 0.0
        self.result_message = ""
        self.result_message_kind = "draw"
        self.result_move_key = None
        
        if self.typewriter_after_id:
            self.root.after_cancel(self.typewriter_after_id)
            self.typewriter_after_id = None
            
        self._refresh_dialog_visibility(False)
        self._refresh_dialog_text(reset_colors=True)
        self.canvas.itemconfig(self.dialog_move_item, image="")
        self.canvas.itemconfig(self.spell_item, image="")
        self.canvas.itemconfig(self.spell_item, state="hidden")

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

        if self.typewriter_after_id is not None:
            try:
                self.root.after_cancel(self.typewriter_after_id)
            except Exception:
                pass
            self.typewriter_after_id = None

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
