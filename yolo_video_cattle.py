#!/usr/bin/env python3
"""
YOLOv8 video analyzer for cattle (conteo + tracking + estimación opcional de área/peso).

Requisitos (instalar en tu PC):
    pip install ultralytics opencv-python numpy

Uso típico (COCO, clase 'cow' autodetectada):
    python yolo_video_cattle.py --source ruta/al/video.mp4 --outdir salida/

Con GPU (si tenés CUDA):
    python yolo_video_cattle.py --source video.mp4 --device 0

Si tenés un modelo propio entrenado (ej. "best.pt"):
    python yolo_video_cattle.py --model best.pt --source video.mp4 --classes -1

Filtrar clases manualmente (IDs):
    # En COCO, 'cow' = 20.
    python yolo_video_cattle.py --source video.mp4 --classes 20

Calibración opcional para estimar área en m^2 y peso (muy aproximado):
    # gsd_m_per_px = metros por pixel (desde altura/cámara); k_weight = kg por m^2
    python yolo_video_cattle.py --source video.mp4 --gsd_m_per_px 0.01 --k_weight 250

Salida:
    - Video anotado con boxes, IDs y contador total (outdir/annotated.mp4).
    - CSV con detecciones por frame (outdir/detections.csv).
    - Resumen en consola con el conteo único de animales.
"""
import os
import csv
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 tracking/conteo de ganado en video")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Ruta al modelo YOLOv8 (.pt). Default: yolov8n.pt")
    p.add_argument("--source", type=str, required=True, help="Ruta al video de entrada")
    p.add_argument("--outdir", type=str, default="runs/cattle", help="Directorio de salida")
    p.add_argument("--conf", type=float, default=0.25, help="Confianza mínima")
    p.add_argument("--iou", type=float, default=0.5, help="IOU para NMS")
    p.add_argument("--imgsz", type=int, default=1280, help="Tamaño de imagen para inferencia")
    p.add_argument("--device", type=str, default=None, help="Dispositivo: 'cpu', '0' (GPU 0), etc.")
    p.add_argument("--classes", type=str, default="auto",
                   help="IDs de clases separadas por coma, '-1' = todas, 'auto' = detectar 'cow/cattle' si existe")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker para seguimiento (por defecto ByteTrack)")
    p.add_argument("--gsd_m_per_px", type=float, default=None, help="Metros por pixel (para estimar área)")
    p.add_argument("--k_weight", type=float, default=None, help="Coeficiente kg/m^2 para estimar peso (muy aproximado)")
    return p.parse_args()


def resolve_classes(model, classes_str):
    """
    Devuelve lista de IDs de clases o None (todas).
    'auto': intenta encontrar índices cuyo nombre sea cow/cattle/bull/calf/vaca/toro/ternero.
    '-1': todas las clases.
    'x,y,z': esos IDs.
    """
    if classes_str == "-1":
        return None  # todas
    if classes_str.lower() == "auto":
        # Obtener mapa de nombres
        names = None
        try:
            names = model.model.names  # algunos modelos
        except Exception:
            pass
        if names is None:
            try:
                names = model.names
            except Exception:
                names = None

        if isinstance(names, dict):
            wanted = {"cow", "cattle", "bull", "calf", "vaca", "toro", "ternero", "novillo", "vaquillona"}
            idxs = [i for i, n in names.items() if str(n).strip().lower() in wanted]
            if idxs:
                return idxs
            # Fallback típico COCO (cow = 20)
            return [20]
        else:
            # Fallback
            return [20]
    # caso IDs separados por coma
    try:
        return [int(x) for x in classes_str.split(",") if x.strip() != ""]
    except Exception:
        return None


def draw_box_with_label(frame, xyxy, label, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # fondo del texto
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Preparar CSV
    csv_path = Path(args.outdir) / "detections.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "time_sec", "track_id", "class_id", "class_name", "conf",
        "x1", "y1", "x2", "y2", "w", "h", "area_px", "area_m2", "est_weight_kg"
    ])

    # FPS del video (para timestamp)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {args.source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # VideoWriter para guardar el video anotado
    out_video_path = str(Path(args.outdir) / "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # Cargar modelo
    model = YOLO(args.model)

    # Resolver clases
    classes = resolve_classes(model, args.classes)
    class_names = None
    try:
        class_names = model.model.names
    except Exception:
        class_names = getattr(model, "names", None)

    # Set para conteo único de animales
    seen_ids = set()

    # Tracking con stream para procesar frame por frame
    frame_idx = 0
    t0 = time.time()

    # Nota: persist=True mantiene el estado del tracker a través de frames
    for result in model.track(
        source=args.source,
        stream=True,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        tracker=args.tracker,
        persist=True,
        classes=classes
    ):
        frame = result.orig_img.copy()
        # resultados
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.full((len(boxes),), -1)
            conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(boxes),))
            ids = boxes.id.cpu().numpy() if boxes.id is not None else np.full((len(boxes),), -1)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                area_px = float(w * h)

                cls_id = int(cls[i]) if cls is not None else -1
                name = str(class_names.get(cls_id, cls_id)) if isinstance(class_names, dict) else str(cls_id)
                score = float(conf[i])
                tid = int(ids[i]) if ids is not None else -1

                # Conteo único
                if tid >= 0:
                    seen_ids.add(tid)

                # Estimaciones opcionales de área/peso
                area_m2 = args.gsd_m_per_px**2 * area_px if args.gsd_m_per_px else None
                est_weight = (area_m2 * args.k_weight) if (area_m2 is not None and args.k_weight) else None

                # Dibujar
                label = f"ID:{tid} {name} {score:.2f}"
                draw_box_with_label(frame, (x1, y1, x2, y2), label)

                # Escribir CSV
                time_sec = frame_idx / fps
                csv_writer.writerow([
                    frame_idx, f"{time_sec:.3f}", tid, cls_id, name, f"{score:.3f}",
                    int(x1), int(y1), int(x2), int(y2), int(w), int(h),
                    f"{area_px:.2f}", f"{area_m2:.4f}" if area_m2 is not None else "",
                    f"{est_weight:.2f}" if est_weight is not None else ""
                ])

        # Overlay: conteo total único
        cv2.putText(frame, f"Cabezas únicas: {len(seen_ids)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"Cabezas únicas: {len(seen_ids)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    writer.release()
    csv_file.close()

    dt = time.time() - t0
    print(f"[OK] Video anotado: {out_video_path}")
    print(f"[OK] CSV de detecciones: {csv_path}")
    print(f"[OK] Conteo único total: {len(seen_ids)}")
    print(f"[INFO] Procesado en {dt:.1f} s")

if __name__ == "__main__":
    main()
