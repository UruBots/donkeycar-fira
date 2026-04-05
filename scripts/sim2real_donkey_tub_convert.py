#!/usr/bin/env python3
"""
Convertidor: Donkey Simulator tub → Formato FIRA Donkeycar

Convierte un tub grabado en Donkey Simulator al formato de runs que usa
el entrenamiento en donkeycar-fira-2024. Redimensiona, recorta y normaliza
imágenes para coincidir con la resolución del entrenamiento / inferencia.

Uso:
    python sim2real_donkey_tub_convert.py \\
        --donkey-tub /ruta/al/tub/del/sim \\
        --dataset-root ../data \\
        --img-w 160 --img-h 120 \\
        --prefix run_sim_

Requisitos:
    - pillow (pip install pillow)
    - numpy
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
from PIL import Image
from io import BytesIO
import base64

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class DonkeySimTubConverter:
    """Convierte un tub del Donkey Sim a formato run de donkeycar-fira"""

    # Formatos soportados del Donkey Sim
    FORMAT_RECORDS_JSON = "records_json"        # records.json con imágenes base64
    FORMAT_RECORD_FILES = "record_files"        # record_*.json + imágenes separadas
    FORMAT_TUB_MANIFEST = "tub_manifest"        # manifest.json + catalog + imágenes

    def __init__(self,
                 donkey_tub_path: str,
                 dataset_root: str,
                 img_w: int = 160,
                 img_h: int = 120,
                 roi_crop_top: int = 0,
                 prefix: str = "run_sim_"):
        """
        Args:
            donkey_tub_path: Ruta a la carpeta del tub del simulador
            dataset_root: Ruta root donde guardar el run convertido
            img_w: Ancho destino de imagen (default 160)
            img_h: Alto destino de imagen (default 120)
            roi_crop_top: Píxeles a recortar parte superior (default 0 = sin recorte)
            prefix: Prefijo para el nombre del run (default "run_sim_")
        """
        self.donkey_tub_path = Path(donkey_tub_path)
        self.dataset_root = Path(dataset_root)
        self.img_w = img_w
        self.img_h = img_h
        self.roi_crop_top = roi_crop_top
        self.prefix = prefix

        if not self.donkey_tub_path.exists():
            raise ValueError(f"Tub path does not exist: {self.donkey_tub_path}")

        self.dataset_root.mkdir(parents=True, exist_ok=True)

        # Generar nombre del run con timestamp
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{prefix}{now}"
        self.output_path = self.dataset_root / self.run_name
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.images_path = self.output_path / "images"
        self.images_path.mkdir(exist_ok=True)

        logger.info(f"Output run: {self.run_name}")
        logger.info(f"Target resolution: {self.img_w}x{self.img_h}")
        if self.roi_crop_top > 0:
            logger.info(f"ROI crop top: {self.roi_crop_top}px")

    def detect_format(self) -> str:
        """Detecta qué formato tiene el tub del simulador"""
        if (self.donkey_tub_path / "records.json").exists():
            logger.info("Detected format: records.json (base64 images)")
            return self.FORMAT_RECORDS_JSON

        record_files = list(self.donkey_tub_path.glob("record_*.json"))
        if record_files:
            logger.info("Detected format: record_*.json (separate image files)")
            return self.FORMAT_RECORD_FILES

        if (self.donkey_tub_path / "manifest.json").exists():
            logger.info("Detected format: tub manifest (catalog + images)")
            return self.FORMAT_TUB_MANIFEST

        raise ValueError(
            f"Unknown tub format. Expected records.json, record_*.json, or manifest.json"
        )

    def load_records_json(self) -> List[Dict]:
        """Lee records.json (imágenes como base64)"""
        records_path = self.donkey_tub_path / "records.json"
        records = []

        try:
            with open(records_path, 'r') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing records.json: {e}")
            return []

        logger.info(f"Loaded {len(records)} records from records.json")
        return records

    def load_record_files(self) -> List[Dict]:
        """Lee record_*.json con imágenes separadas"""
        record_files = sorted(self.donkey_tub_path.glob("record_*.json"))
        records = []

        for record_file in record_files:
            try:
                with open(record_file, 'r') as f:
                    record = json.load(f)
                    # Si la imagen está en path (string), cargarla y almacenarla
                    if 'image_array' in record and isinstance(record['image_array'], str):
                        img_path = self.donkey_tub_path / record['image_array']
                        if img_path.exists():
                            # Cargar imagen y almacenarla como base64 o dejarla como path
                            record['image_path'] = str(img_path)
                    records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping {record_file}: {e}")

        logger.info(f"Loaded {len(records)} records from record_*.json")
        return records

    def load_tub_manifest(self) -> List[Dict]:
        """Lee manifest.json con catalog"""
        manifest_path = self.donkey_tub_path / "manifest.json"
        records = []

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # Leer catálogos
            catalog_files = sorted(self.donkey_tub_path.glob("catalog_*.catalog"))
            for catalog_file in catalog_files:
                with open(catalog_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))

            logger.info(f"Loaded {len(records)} records from tub manifest")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading manifest: {e}")

        return records

    def load_records(self) -> List[Dict]:
        """Carga registros en el formato detectado"""
        fmt = self.detect_format()

        if fmt == self.FORMAT_RECORDS_JSON:
            return self.load_records_json()
        elif fmt == self.FORMAT_RECORD_FILES:
            return self.load_record_files()
        elif fmt == self.FORMAT_TUB_MANIFEST:
            return self.load_tub_manifest()

        return []

    def decode_image(self, record: Dict) -> Optional[np.ndarray]:
        """Extrae imagen del registro (base64 o path)"""
        # Intento 1: image_array como base64 string
        if 'image_array' in record and isinstance(record['image_array'], str):
            try:
                img_data = base64.b64decode(record['image_array'])
                img = Image.open(BytesIO(img_data)).convert('RGB')
                return np.array(img)
            except Exception as e:
                logger.debug(f"Failed to decode base64 image: {e}")

        # Intento 2: image_path (ruta a archivo)
        if 'image_path' in record:
            try:
                img = Image.open(record['image_path']).convert('RGB')
                return np.array(img)
            except Exception as e:
                logger.debug(f"Failed to load image from path: {e}")

        # Intento 3: image_file en registro (ruta relativa)
        if 'image_file' in record:
            img_path = self.donkey_tub_path / record['image_file']
            try:
                img = Image.open(img_path).convert('RGB')
                return np.array(img)
            except Exception as e:
                logger.debug(f"Failed to load {img_path}: {e}")

        # Intento 4: buscar por índice (0.jpg, 1.jpg, ...)
        if '_index' in record:
            img_path = self.donkey_tub_path / f"{record['_index']}.jpg"
            try:
                img = Image.open(img_path).convert('RGB')
                return np.array(img)
            except Exception:
                pass

        return None

    def resize_and_crop(self, img: np.ndarray) -> np.ndarray:
        """Redimensiona y recorta imagen al tamaño objetivo"""
        h, w = img.shape[:2]

        # Aplicar ROI crop en parte superior si es necesario
        if self.roi_crop_top > 0 and h > self.roi_crop_top:
            img = img[self.roi_crop_top:, :]

        # Redimensionar a tamaño objetivo
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((self.img_w, self.img_h), Image.Resampling.LANCZOS)
        return np.array(pil_img)

    def extract_controls(self, record: Dict) -> Tuple[float, float]:
        """Extrae angle y throttle del registro"""
        # Buscar variantes comunes de nombres de controles
        angle_keys = ['user/angle', 'angle', 'steering_angle', 'steer']
        throttle_keys = ['user/throttle', 'throttle', 'motor/speed', 'speed']

        angle = 0.0
        for key in angle_keys:
            if key in record:
                angle = float(record[key])
                break

        throttle = 0.0
        for key in throttle_keys:
            if key in record:
                throttle = float(record[key])
                break

        # Normalizar throttle a [-1, 1] si está en [0, 5] (rango Gym)
        if throttle > 1.5:
            throttle = (throttle - 2.5) / 2.5  # Mapear [0, 5] → [-1, 1]

        return float(angle), float(throttle)

    def convert(self) -> int:
        """Realiza la conversión y retorna número de registros procesados"""
        records = self.load_records()

        if not records:
            logger.error("No records loaded from tub")
            return 0

        manifest_records = []
        image_idx = 0
        skipped = 0

        for i, record in enumerate(records):
            img = self.decode_image(record)
            if img is None:
                logger.debug(f"Skipping record {i}: could not decode image")
                skipped += 1
                continue

            # Redimensionar y recortar
            img = self.resize_and_crop(img)

            # Guardar imagen
            img_filename = f"{image_idx}.jpg"
            img_path = self.images_path / img_filename
            Image.fromarray(img).save(img_path, quality=95)

            # Extraer controles
            angle, throttle = self.extract_controls(record)

            # Crear registro FIRA format
            manifest_record = {
                'image_array': img_filename,
                'user/angle': angle,
                'user/throttle': throttle,
            }

            # Copiar metadata adicional si existe
            if 'timestamp' in record:
                manifest_record['timestamp'] = record['timestamp']

            manifest_records.append(manifest_record)
            image_idx += 1

            if (image_idx) % 50 == 0:
                logger.info(f"Processed {image_idx} images...")

        # Guardar manifest.json
        manifest_path = self.output_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                'version': 1,
                'type': 'tub_v2',
                'records': image_idx,
                'source': str(self.donkey_tub_path),
                'created': datetime.now().isoformat(),
                'img_w': self.img_w,
                'img_h': self.img_h,
                'roi_crop_top': self.roi_crop_top,
            }, f, indent=2)

        # Guardar catalog (formato FIRA)
        catalog_path = self.output_path / "catalog_0.catalog"
        with open(catalog_path, 'w') as f:
            for record in manifest_records:
                f.write(json.dumps(record) + '\n')

        logger.info(f"\n✓ Conversion complete!")
        logger.info(f"  Processed: {image_idx} images")
        logger.info(f"  Skipped: {skipped} records")
        logger.info(f"  Output: {self.output_path}")
        logger.info(f"  Use in training: --tub-paths {self.output_path}")

        return image_idx


def main():
    parser = argparse.ArgumentParser(
        description='Convert Donkey Simulator tub to FIRA Donkeycar format'
    )
    parser.add_argument('--donkey-tub', required=True,
                        help='Path to Donkey Simulator tub folder')
    parser.add_argument('--dataset-root', default='./data',
                        help='Root folder for output runs (default: ./data)')
    parser.add_argument('--img-w', type=int, default=160,
                        help='Target image width (default: 160)')
    parser.add_argument('--img-h', type=int, default=120,
                        help='Target image height (default: 120)')
    parser.add_argument('--roi-crop-top', type=int, default=0,
                        help='Pixels to crop from top (default: 0)')
    parser.add_argument('--prefix', default='run_sim_',
                        help='Prefix for run name (default: run_sim_)')

    args = parser.parse_args()

    converter = DonkeySimTubConverter(
        donkey_tub_path=args.donkey_tub,
        dataset_root=args.dataset_root,
        img_w=args.img_w,
        img_h=args.img_h,
        roi_crop_top=args.roi_crop_top,
        prefix=args.prefix
    )

    num_records = converter.convert()
    return 0 if num_records > 0 else 1


if __name__ == '__main__':
    exit(main())
