#!/usr/bin/env python3
"""
TP1 - Transformada de Viewport (Python + tkinter)

Como usar:
    - Coloque este arquivo no mesmo diretório do seu `entrada.xml` ou
      rode passando o caminho:
        python tp1.py --input entrada.xml --output saida.xml
    - A janela abrirá com a viewport renderizada e um minimapa.
    - Use as teclas de seta para mover a WINDOW (1 unidade por pressionamento).
    - Clique em "Salvar XML de saída" para gerar um arquivo XML com as
      coordenadas transformadas (viewport) adicionadas.

Saída padrão: saida.xml
"""

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import copy
from pathlib import Path

# -------------------------------
# Data classes for the model
# -------------------------------

@dataclass
class Point2D:
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

@dataclass
class Line2D:
    p1: Point2D
    p2: Point2D

@dataclass
class Polygon2D:
    points: List[Point2D]

@dataclass
class Window:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def move(self, dx: float, dy: float):
        self.xmin += dx
        self.xmax += dx
        self.ymin += dy
        self.ymax += dy

    def width(self) -> float:
        return self.xmax - self.xmin

    def height(self) -> float:
        return self.ymax - self.ymin

@dataclass
class Viewport:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def width(self) -> float:
        return self.xmax - self.xmin

    def height(self) -> float:
        return self.ymax - self.ymin

@dataclass
class Scene:
    viewport: Viewport
    window: Window
    pontos: List[Point2D] = field(default_factory=list)
    retas: List[Line2D] = field(default_factory=list)
    poligonos: List[Polygon2D] = field(default_factory=list)

# -------------------------------
# Utilities: matrix (3x3 homogeneous)
# -------------------------------

def make_viewport_matrix(window: Window, viewport: Viewport) -> List[List[float]]:
    """
    Constrói a matriz homogênea 3x3 que transforma (xw, yw, 1) -> (xv, yv, 1)
    levando em conta a inversão do eixo Y (origem no canto superior esquerdo).
    Formula:
        sx = (vxmax - vxmin) / (wxmax - wxmin)
        sy = (vymax - vymin) / (wymax - wymin)
        xvp = vxmin + (xw - wxmin) * sx
        yvp = vymax - (yw - wymin) * sy  (inversão)
    Matrizes combinadas em homogênea para:
      - translação de referência
      - escala
      - inversão Y
    """
    wxmin, wymin, wxmax, wymax = window.xmin, window.ymin, window.xmax, window.ymax
    vxmin, vymin, vxmax, vymax = viewport.xmin, viewport.ymin, viewport.xmax, viewport.ymax
    sx = (vxmax - vxmin) / (wxmax - wxmin)
    sy = (vymax - vymin) / (wymax - wymin)

    # We'll build a single matrix M such that [xv, yv, 1]^T = M * [xw, yw, 1]^T
    # Approach:
    # 1) Translate world so that wxmin,wymin -> 0,0  (T1)
    # 2) Scale by sx, sy (S)
    # 3) Flip y and translate to viewport coords: yvp = vymax - sy*(yw - wymin)
    #    i.e. after scaling, y' = sy*(yw - wymin); to get yvp = vymax - y' => translate by vymax and scale by -1 in y
    # Equivalent single matrix:
    # M = [ sx,  0, vxmin - sx*wxmin ]
    #     [ 0, -sy, vymax + sy*wymin ]
    #     [ 0,  0, 1 ]
    tx = vxmin - sx * wxmin
    ty = vymax + sy * wymin  # note plus because of sign flip in scale on y

    M = [
        [sx, 0.0, tx],
        [0.0, -sy, ty],
        [0.0, 0.0, 1.0]
    ]
    return M

def apply_matrix(M: List[List[float]], p: Point2D) -> Point2D:
    xw, yw = p.x, p.y
    xv = M[0][0] * xw + M[0][1] * yw + M[0][2]
    yv = M[1][0] * xw + M[1][1] * yw + M[1][2]
    return Point2D(xv, yv)

# -------------------------------
# XML parsing / writing
# -------------------------------

def parse_input_xml(path: str) -> Scene:
    tree = ET.parse(path)
    root = tree.getroot()

    # parse viewport
    vp = root.find('viewport')
    if vp is None:
        raise ValueError("Arquivo XML não contém elemento <viewport>.")
    vpmin = vp.find('vpmin')
    vpmax = vp.find('vpmax')
    viewport = Viewport(
        xmin=float(vpmin.get('x')),
        ymin=float(vpmin.get('y')),
        xmax=float(vpmax.get('x')),
        ymax=float(vpmax.get('y'))
    )

    # parse window
    w = root.find('window')
    if w is None:
        raise ValueError("Arquivo XML não contém elemento <window>.")
    wmin = w.find('wmin')
    wmax = w.find('wmax')
    window = Window(
        xmin=float(wmin.get('x')),
        ymin=float(wmin.get('y')),
        xmax=float(wmax.get('x')),
        ymax=float(wmax.get('y'))
    )

    # parse objects
    pontos = []
    retas = []
    poligonos = []
    # root may contain top-level <ponto> elements and children <reta>, <poligono>
    for child in root:
        if child.tag == 'ponto':
            x = float(child.get('x'))
            y = float(child.get('y'))
            pontos.append(Point2D(x, y))
        elif child.tag == 'reta':
            pts = child.findall('ponto')
            if len(pts) != 2:
                raise ValueError("Elemento <reta> deve conter exatamente 2 <ponto>.")
            p1 = Point2D(float(pts[0].get('x')), float(pts[0].get('y')))
            p2 = Point2D(float(pts[1].get('x')), float(pts[1].get('y')))
            retas.append(Line2D(p1, p2))
        elif child.tag == 'poligono':
            pts = child.findall('ponto')
            poly_pts = [Point2D(float(p.get('x')), float(p.get('y'))) for p in pts]
            poligonos.append(Polygon2D(poly_pts))
        else:
            # ignore other tags (viewport/window already handled)
            pass

    return Scene(viewport=viewport, window=window, pontos=pontos, retas=retas, poligonos=poligonos)

def write_output_xml(input_path: str, output_path: str, scene: Scene, transformed: Dict):
    """
    Gera um XML similar ao de entrada, mas adicionando os pontos em coordenadas de viewport.
    'transformed' deve conter listas de pontos transformados com as mesmas chaves: 'pontos', 'retas', 'poligonos'
    """
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Remove previous 'viewport_coords' if any
    existing = root.find('viewport_coords')
    if existing is not None:
        root.remove(existing)

    vc = ET.Element('viewport_coords')

    # pontos
    for p in transformed.get('pontos', []):
        pe = ET.Element('ponto_vp')
        pe.set('x', f"{p.x:.6f}")
        pe.set('y', f"{p.y:.6f}")
        vc.append(pe)

    # retas
    for r in transformed.get('retas', []):
        re_e = ET.Element('reta_vp')
        p1e = ET.Element('ponto')
        p1e.set('x', f"{r.p1.x:.6f}")
        p1e.set('y', f"{r.p1.y:.6f}")
        p2e = ET.Element('ponto')
        p2e.set('x', f"{r.p2.x:.6f}")
        p2e.set('y', f"{r.p2.y:.6f}")
        re_e.append(p1e)
        re_e.append(p2e)
        vc.append(re_e)

    # poligonos
    for poly in transformed.get('poligonos', []):
        poly_e = ET.Element('poligono_vp')
        for pt in poly.points:
            pte = ET.Element('ponto')
            pte.set('x', f"{pt.x:.6f}")
            pte.set('y', f"{pt.y:.6f}")
            poly_e.append(pte)
        vc.append(poly_e)

    root.append(vc)

    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Arquivo de saída salvo em: {output_path}")

# -------------------------------
# GUI / Rendering
# -------------------------------

class TP1App:
    def __init__(self, master: tk.Tk, scene: Scene, input_xml_path: str, output_xml_path: str):
        self.master = master
        self.scene = scene
        self.input_xml_path = input_xml_path
        self.output_xml_path = output_xml_path

        # window title & geometry determined by viewport size
        vpw = int(scene.viewport.width())
        vph = int(scene.viewport.height())

        master.title("TP1 - Transformada de Viewport")
        # Create frames
        self.mainframe = ttk.Frame(master, padding="6")
        self.mainframe.grid(row=0, column=0, sticky='nsew')
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)

        # Canvas for viewport (primary)
        self.canvas = tk.Canvas(self.mainframe, width=vpw, height=vph, bg='white')
        self.canvas.grid(row=0, column=0, sticky='nsew')
        # Minimap: a small canvas
        self.minimap_size = 200
        self.minimap = tk.Canvas(self.mainframe, width=self.minimap_size, height=self.minimap_size, bg='#eee')
        self.minimap.grid(row=0, column=1, padx=8, pady=4, sticky='ne')

        # Controls
        self.controls = ttk.Frame(self.mainframe)
        self.controls.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(6,0))
        self.save_btn = ttk.Button(self.controls, text="Salvar XML de saída", command=self.save_output)
        self.save_btn.pack(side='left')
        self.help_label = ttk.Label(self.controls, text="Use setas para mover a window (1.0 unidade).")
        self.help_label.pack(side='left', padx=10)

        # Make canvas expand
        self.mainframe.rowconfigure(0, weight=1)
        self.mainframe.columnconfigure(0, weight=1)

        # keyboard bindings
        master.bind("<Left>", lambda e: self.on_move(-1.0, 0.0))
        master.bind("<Right>", lambda e: self.on_move(1.0, 0.0))
        master.bind("<Up>", lambda e: self.on_move(0.0, -1.0))
        master.bind("<Down>", lambda e: self.on_move(0.0, 1.0))
        master.bind("<Escape>", lambda e: master.quit())

        # initial transform & draw
        self.redraw()

    def compute_transformed(self) -> Dict:
        """
        Calcula as coordenadas transformadas (viewport) de todos os objetos
        e retorna em um dicionário com as mesmas chaves.
        """
        M = make_viewport_matrix(self.scene.window, self.scene.viewport)
        t_pontos = [apply_matrix(M, p) for p in self.scene.pontos]
        t_retas = [Line2D(apply_matrix(M, r.p1), apply_matrix(M, r.p2)) for r in self.scene.retas]
        t_poligonos = [Polygon2D([apply_matrix(M, p) for p in poly.points]) for poly in self.scene.poligonos]
        return {'pontos': t_pontos, 'retas': t_retas, 'poligonos': t_poligonos}

    def redraw(self):
        # Clear
        self.canvas.delete('all')
        self.minimap.delete('all')

        transformed = self.compute_transformed()

        # Draw polygons (wireframe)
        for poly in transformed['poligonos']:
            if len(poly.points) >= 2:
                coords = []
                for p in poly.points:
                    coords.extend([p.x, p.y])
                # close polygon
                coords.extend([poly.points[0].x, poly.points[0].y])
                self.canvas.create_line(*coords, width=2)
        # Draw lines
        for r in transformed['retas']:
            self.canvas.create_line(r.p1.x, r.p1.y, r.p2.x, r.p2.y, width=2, dash=(1,0))
        # Draw points
        for p in transformed['pontos']:
            self._draw_point_on_canvas(self.canvas, p)

        # Draw minimap
        self.draw_minimap()

    def _draw_point_on_canvas(self, canvas: tk.Canvas, p: Point2D, r: float = 3.5):
        x, y = p.x, p.y
        canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')

    def draw_minimap(self):
        """Desenha o minimapa: todo o mundo (window total bounding) e a window atual."""
        # We'll build a bounding box for the "world" from objects and maybe window extents.
        # Simpler: use the window extents AND objects to get world bounding box.
        all_x = []
        all_y = []
        for p in self.scene.pontos:
            all_x.append(p.x); all_y.append(p.y)
        for r in self.scene.retas:
            all_x.extend([r.p1.x, r.p2.x]); all_y.extend([r.p1.y, r.p2.y])
        for poly in self.scene.poligonos:
            for p in poly.points:
                all_x.append(p.x); all_y.append(p.y)
        # include window extents too, to ensure visible
        all_x.extend([self.scene.window.xmin, self.scene.window.xmax])
        all_y.extend([self.scene.window.ymin, self.scene.window.ymax])

        if not all_x or not all_y:
            return

        wxmin, wxmax = min(all_x), max(all_x)
        wymin, wymax = min(all_y), max(all_y)
        # pad slightly
        pad_x = (wxmax - wxmin) * 0.05 if wxmax > wxmin else 1.0
        pad_y = (wymax - wymin) * 0.05 if wymax > wymin else 1.0
        wxmin -= pad_x; wxmax += pad_x
        wymin -= pad_y; wymax += pad_y

        W = self.minimap_size
        H = self.minimap_size

        def world_to_minimap(px, py):
            sx = W / (wxmax - wxmin)
            sy = H / (wymax - wymin)
            # keep aspect ratio by using smaller scale
            s = min(sx, sy)
            mx = (px - wxmin) * s
            my = H - (py - wymin) * s  # invert y for canvas coordinates
            # center if aspect mismatch
            # compute margins
            tx = (W - (wxmax - wxmin) * s) / 2
            ty = (H - (wymax - wymin) * s) / 2
            return mx + tx, my - ty

        # Draw all objects in minimap (simple)
        # polygons
        for poly in self.scene.poligonos:
            coords = []
            for p in poly.points:
                mx, my = world_to_minimap(p.x, p.y)
                coords.extend([mx, my])
            # close polygon
            if coords:
                coords.extend(coords[:2])
                self.minimap.create_line(*coords, width=1)
        # retas
        for r in self.scene.retas:
            p1 = world_to_minimap(r.p1.x, r.p1.y)
            p2 = world_to_minimap(r.p2.x, r.p2.y)
            self.minimap.create_line(p1[0], p1[1], p2[0], p2[1], width=1)
        # pontos
        for p in self.scene.pontos:
            mx, my = world_to_minimap(p.x, p.y)
            self.minimap.create_oval(mx-2, my-2, mx+2, my+2, fill='black')

        # draw outer world box
        # outer rectangle (for visualization)
        # draw current window rectangle
        wx1, wy1 = world_to_minimap(self.scene.window.xmin, self.scene.window.ymin)
        wx2, wy2 = world_to_minimap(self.scene.window.xmax, self.scene.window.ymax)
        # Because world_to_minimap inverts y and we applied offsets, coords may be swapped
        self.minimap.create_rectangle(wx1, wy1, wx2, wy2, outline='red', width=2)

    def on_move(self, dx: float, dy: float):
        # Move the window; note: positive dy -> move window up in world? The assignment defined that pressing arrow moves window in that direction.
        # Here we move window straightforwardly: Up arrow moved window by dy=-1.0, per binding.
        self.scene.window.move(dx, dy)
        # redraw
        self.redraw()

    def save_output(self):
        # compute transformed and write
        transformed = self.compute_transformed()
        # ask for file
        out_path = self.output_xml_path
        # if given path exists, just overwrite; otherwise ask
        try:
            write_output_xml(self.input_xml_path, out_path, self.scene, transformed)
            messagebox.showinfo("Salvar", f"Arquivo salvo: {out_path}")
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível salvar o arquivo: {e}")

# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="TP1 - Transformada de Viewport")
    parser.add_argument('--input', '-i', default='entrada.xml', help='Arquivo XML de entrada')
    parser.add_argument('--output', '-o', default='saida.xml', help='Arquivo XML de saída')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not Path(input_path).exists():
        print(f"Arquivo de entrada '{input_path}' não encontrado.")
        return

    scene = parse_input_xml(input_path)

    # Create Tkinter window and app
    root = tk.Tk()
    app = TP1App(root, scene, input_path, output_path)
    root.mainloop()

if __name__ == "__main__":
    main()
