# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""The subset of the pyimgui API used by the visualizer, implemented on top of imgui-bundle.

The visualizer was written for pyimgui 1.3, which wraps Dear ImGui 1.7x and no longer installs on
recent Python versions (there are no wheels, and the sources do not build). imgui-bundle is
maintained and ships wheels for Linux, Windows and macOS. Importing this module as `imgui` lets
the widgets keep their pyimgui style calls."""

import numpy as np
from imgui_bundle import imgui as _imgui

ImVec2 = _imgui.ImVec2
ImVec4 = _imgui.ImVec4

#----------------------------------------------------------------------------
# Constants.

_Col = _imgui.Col_
COLOR_TEXT                      = _Col.text.value
COLOR_TEXT_DISABLED             = _Col.text_disabled.value
COLOR_WINDOW_BACKGROUND         = _Col.window_bg.value
COLOR_CHILD_BACKGROUND          = _Col.child_bg.value
COLOR_POPUP_BACKGROUND          = _Col.popup_bg.value
COLOR_BORDER                    = _Col.border.value
COLOR_FRAME_BACKGROUND          = _Col.frame_bg.value
COLOR_FRAME_BACKGROUND_HOVERED  = _Col.frame_bg_hovered.value
COLOR_FRAME_BACKGROUND_ACTIVE   = _Col.frame_bg_active.value
COLOR_MENUBAR_BACKGROUND        = _Col.menu_bar_bg.value
COLOR_SCROLLBAR_GRAB            = _Col.scrollbar_grab.value
COLOR_CHECK_MARK                = _Col.check_mark.value
COLOR_SLIDER_GRAB               = _Col.slider_grab.value
COLOR_SLIDER_GRAB_ACTIVE        = _Col.slider_grab_active.value
COLOR_BUTTON                    = _Col.button.value
COLOR_BUTTON_HOVERED            = _Col.button_hovered.value
COLOR_BUTTON_ACTIVE             = _Col.button_active.value
COLOR_HEADER                    = _Col.header.value
COLOR_HEADER_HOVERED            = _Col.header_hovered.value
COLOR_HEADER_ACTIVE             = _Col.header_active.value

WINDOW_NO_TITLE_BAR                 = _imgui.WindowFlags_.no_title_bar.value
WINDOW_NO_RESIZE                    = _imgui.WindowFlags_.no_resize.value
WINDOW_NO_MOVE                      = _imgui.WindowFlags_.no_move.value
WINDOW_ALWAYS_VERTICAL_SCROLLBAR    = _imgui.WindowFlags_.always_vertical_scrollbar.value

TREE_NODE_LEAF                      = _imgui.TreeNodeFlags_.leaf.value
TREE_NODE_DEFAULT_OPEN              = _imgui.TreeNodeFlags_.default_open.value

INPUT_TEXT_ENTER_RETURNS_TRUE       = _imgui.InputTextFlags_.enter_returns_true.value
INPUT_TEXT_AUTO_SELECT_ALL          = _imgui.InputTextFlags_.auto_select_all.value

DRAG_DROP_SOURCE_NO_PREVIEW_TOOLTIP = _imgui.DragDropFlags_.source_no_preview_tooltip.value

STYLE_FRAME_PADDING                 = _imgui.StyleVar_.frame_padding.value

_FLT_MAX = 3.4028234663852886e+38

#----------------------------------------------------------------------------
# Context, frames, IO.

create_context      = _imgui.create_context
destroy_context     = _imgui.destroy_context
get_io              = _imgui.get_io
new_frame           = _imgui.new_frame
render              = _imgui.render
end_frame           = _imgui.end_frame
get_draw_data       = _imgui.get_draw_data

def _unwrap_style(style):
    return style._style if isinstance(style, _Style) else style

def style_colors_dark(dst=None):
    _imgui.style_colors_dark(_unwrap_style(dst))

def style_colors_light(dst=None):
    _imgui.style_colors_light(_unwrap_style(dst))

def style_colors_classic(dst=None):
    _imgui.style_colors_classic(_unwrap_style(dst))

def push_font(font, size=0.0):
    _imgui.push_font(font, float(size))

pop_font = _imgui.pop_font

#----------------------------------------------------------------------------
# Style.

class _StyleColors:
    def __init__(self, style):
        self._style = style
    def __getitem__(self, idx):
        c = self._style.color_(idx)
        return (c.x, c.y, c.z, c.w)
    def __setitem__(self, idx, value):
        self._style.set_color_(idx, ImVec4(*value))

class _Style:
    """pyimgui style: vectors as lists and `colors[idx]` as RGBA tuples."""
    def __init__(self, style):
        object.__setattr__(self, '_style', style)
    @property
    def colors(self):
        return _StyleColors(self._style)
    def __getattr__(self, name):
        return getattr(self._style, name)
    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            value = ImVec2(*value)
        setattr(self._style, name, value)

def get_style():
    return _Style(_imgui.get_style())

def push_style_color(idx, r, g, b, a=1.0):
    _imgui.push_style_color(idx, ImVec4(r, g, b, a))

def pop_style_color(count=1):
    _imgui.pop_style_color(count)

def push_style_var(idx, value):
    _imgui.push_style_var(idx, ImVec2(*value) if isinstance(value, (list, tuple)) else float(value))

def pop_style_var(count=1):
    _imgui.pop_style_var(count)

push_item_width = _imgui.push_item_width
pop_item_width  = _imgui.pop_item_width

def push_id(str_id):
    _imgui.push_id(str(str_id))

pop_id = _imgui.pop_id

#----------------------------------------------------------------------------
# Windows and layout.

def begin(label, closable=False, flags=0):
    return _imgui.begin(label, True if closable else None, flags)

end = _imgui.end

def begin_child(label, width=0, height=0, border=False, flags=0):
    child_flags = _imgui.ChildFlags_.borders.value if border else 0
    return _imgui.begin_child(label, ImVec2(width, height), child_flags, flags)

end_child = _imgui.end_child

def set_next_window_position(x, y, condition=0):
    _imgui.set_next_window_pos(ImVec2(x, y), condition)

def set_next_window_size(width, height, condition=0):
    _imgui.set_next_window_size(ImVec2(width, height), condition)

def same_line(position=0.0, spacing=-1.0):
    _imgui.same_line(float(position), float(spacing))

def get_content_region_max():
    # GetContentRegionMax() was removed in Dear ImGui 1.92, this is its documented replacement (in window coordinates).
    avail = _imgui.get_content_region_avail()
    cursor = _imgui.get_cursor_screen_pos()
    window = _imgui.get_window_pos()
    return (avail.x + cursor.x - window.x, avail.y + cursor.y - window.y)

get_cursor_pos_x    = _imgui.get_cursor_pos_x
get_cursor_pos_y    = _imgui.get_cursor_pos_y
set_cursor_pos_x    = _imgui.set_cursor_pos_x
set_cursor_pos_y    = _imgui.set_cursor_pos_y
get_text_line_height = _imgui.get_text_line_height
get_text_line_height_with_spacing = _imgui.get_text_line_height_with_spacing

def set_scroll_here(center=0.5):
    _imgui.set_scroll_here_y(center)

#----------------------------------------------------------------------------
# Widgets.

def text(text):
    _imgui.text_unformatted(str(text)) # no printf style formatting of '%' in the text

def text_colored(text, r, g, b, a=1.0):
    _imgui.push_style_color(COLOR_TEXT, ImVec4(r, g, b, a))
    _imgui.text_unformatted(str(text))
    _imgui.pop_style_color(1)

def set_tooltip(text):
    if _imgui.begin_tooltip():
        _imgui.text_unformatted(str(text))
        _imgui.end_tooltip()

def button(label, width=0, height=0):
    return _imgui.button(label, ImVec2(width, height))

def checkbox(label, state):
    return _imgui.checkbox(label, bool(state))

def selectable(label, selected=False, flags=0, width=0, height=0):
    return _imgui.selectable(label, bool(selected), flags, ImVec2(width, height))

def collapsing_header(text, visible=None, flags=0):
    if visible is None:
        return _imgui.collapsing_header(text, flags), None
    return _imgui.collapsing_header(text, bool(visible), flags)

def _slider_flags(power):
    # pyimgui's power curves were replaced by logarithmic sliders in Dear ImGui 1.78
    return _imgui.SliderFlags_.logarithmic.value if power != 1 else 0

def slider_float(label, value, min_value, max_value, format='%.3f', power=1.0):
    return _imgui.slider_float(label, float(value), float(min_value), float(max_value), format, _slider_flags(power))

def slider_int(label, value, min_value, max_value, format='%d'):
    return _imgui.slider_int(label, int(value), int(min_value), int(max_value), format)

def drag_int(label, value, change_speed=1.0, min_value=0, max_value=0, format='%d'):
    return _imgui.drag_int(label, int(value), float(change_speed), int(min_value), int(max_value), format)

def input_int(label, value, step=1, step_fast=100, flags=0):
    return _imgui.input_int(label, int(value), step, step_fast, flags)

def input_float(label, value, step=0.0, step_fast=0.0, format='%.3f', flags=0):
    return _imgui.input_float(label, float(value), step, step_fast, format, flags)

def input_float2(label, value0, value1, format='%.3f', flags=0):
    changed, values = _imgui.input_float2(label, [float(value0), float(value1)], format, flags)
    return changed, (values[0], values[1])

def input_text(label, value, buffer_length=None, flags=0):
    return _imgui.input_text(label, value, flags)

def plot_lines(label, values, values_offset=0, overlay_text=None, scale_min=_FLT_MAX, scale_max=_FLT_MAX, graph_size=(0, 0)):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    _imgui.plot_lines(label, values, values_offset, overlay_text, scale_min, scale_max, ImVec2(*graph_size))

is_item_hovered = _imgui.is_item_hovered
is_item_active  = _imgui.is_item_active

#----------------------------------------------------------------------------
# Popups and menus.

open_popup  = _imgui.open_popup
end_popup   = _imgui.end_popup
end_menu    = _imgui.end_menu

def begin_popup(label, flags=0):
    return _imgui.begin_popup(label, flags)

def begin_menu(label, enabled=True):
    return _imgui.begin_menu(label, enabled)

def menu_item(label, shortcut=None, selected=False, enabled=True):
    return _imgui.menu_item(label, shortcut or '', bool(selected), enabled)

#----------------------------------------------------------------------------
# Mouse dragging.

def begin_drag_drop_source(flags=0):
    return _imgui.begin_drag_drop_source(flags)

end_drag_drop_source = _imgui.end_drag_drop_source

def get_mouse_drag_delta(button=0, lock_threshold=-1.0):
    delta = _imgui.get_mouse_drag_delta(button, lock_threshold)
    return delta.x, delta.y

def reset_mouse_drag_delta(button=0):
    _imgui.reset_mouse_drag_delta(button)

#----------------------------------------------------------------------------
