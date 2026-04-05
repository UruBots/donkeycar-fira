#!/usr/bin/env python3
"""
Entrenador sim2real: Mezcla datos simulados + reales

Entrena un modelo Donkeycar con datos de simulación + datos reales.
Soporta multi-tub training, data augmentation y domain randomization.

Uso básico:
    python train_sim2real.py \\
        --model pilotnet \\
        --tub sim_run_1 sim_run_2 real_run_1 \\
        --epochs 60

Uso con config:
    python train_sim2real.py \\
        --config mycar_config.py \\
        --model pilotnet

Requisitos:
    - donkey-car environment configurado
    - tensorflow >= 2.4
    - numpy, pillow
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List
import logging

# Agregar paths para imports de donkey
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from donkeycar.config import Config
from donkeycar.pipeline.training import train
from donkeycar.utils import get_model_by_type

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class Sim2RealTrainer:
    """Entrena modelo con datos sim + real"""

    def __init__(self, config_module: str = None):
        """Carga configuración"""
        self.cfg = Config()

        if config_module:
            # Importar configuración personalizada
            module_name = Path(config_module).stem
            spec = __import__(module_name)
            for attr in dir(spec):
                if attr.isupper():
                    setattr(self.cfg, attr, getattr(spec, attr))
            logger.info(f"Loaded config from {config_module}")
        else:
            logger.info("Using default Config()")

    def validate_tubs(self, tub_paths: List[str]) -> bool:
        """Valida que los tubs existan"""
        missing = []
        for tub in tub_paths:
            tub_path = Path(tub).expanduser()
            if not tub_path.exists():
                missing.append(tub)
            else:
                logger.info(f"✓ Found tub: {tub_path}")

        if missing:
            logger.error(f"Missing tubs: {missing}")
            return False

        return True

    def summarize_dataset(self, tub_paths: List[str]) -> None:
        """Resume el dataset a entrenar"""
        logger.info("\n" + "="*60)
        logger.info("DATASET SUMMARY")
        logger.info("="*60)

        total_records = 0
        for tub in tub_paths:
            tub_path = Path(tub).expanduser()
            catalog_files = list(tub_path.glob("catalog_*.catalog"))

            if catalog_files:
                # Contar líneas en catálogos (cada línea = 1 registro)
                count = 0
                for catalog_file in catalog_files:
                    with open(catalog_file, 'r') as f:
                        count += sum(1 for _ in f)
                logger.info(f"  {tub_path.name:30s} → {count:6d} records")
                total_records += count
            else:
                logger.warning(f"  {tub_path.name:30s} → (no catalog found)")

        logger.info("-" * 60)
        logger.info(f"  TOTAL                         → {total_records:6d} records")
        logger.info("="*60 + "\n")

    def train_model(self,
                    model_type: str,
                    tub_paths: List[str],
                    transfer_from: str = None,
                    epochs: int = None,
                    batch_size: int = None) -> str:
        """Entrena el modelo"""

        # Validar tubs
        if not self.validate_tubs(tub_paths):
            raise ValueError("Invalid tub paths")

        # Resumen dataset
        self.summarize_dataset(tub_paths)

        # Actualizar config si se proporcionan parámetros
        if epochs:
            self.cfg.MAX_EPOCHS = epochs
        if batch_size:
            self.cfg.BATCH_SIZE = batch_size

        # Convertir lista a string comma-separated
        tub_paths_str = ','.join(tub_paths)

        logger.info("="*60)
        logger.info("TRAINING CONFIGURATION")
        logger.info("="*60)
        logger.info(f"  Model type:          {model_type}")
        logger.info(f"  Tub paths:           {tub_paths_str}")
        logger.info(f"  Max epochs:          {self.cfg.MAX_EPOCHS}")
        logger.info(f"  Batch size:          {self.cfg.BATCH_SIZE}")
        logger.info(f"  Train/test split:    {self.cfg.TRAIN_TEST_SPLIT}")

        # Domain randomization info
        if hasattr(self.cfg, 'AUGMENTATIONS'):
            logger.info(f"  Augmentations:       {len(self.cfg.AUGMENTATIONS)} types")
        if hasattr(self.cfg, 'TRANSFORMATIONS'):
            logger.info(f"  Transformations:     {len(self.cfg.TRANSFORMATIONS)} types")

        if transfer_from:
            logger.info(f"  Transfer from:       {transfer_from}")

        logger.info("="*60 + "\n")

        # Entrenar
        try:
            logger.info("Starting training...")
            model_path = train(
                cfg=self.cfg,
                tub_paths=tub_paths_str,
                model=transfer_from,
                model_type=model_type,
                comment="sim2real mixed training"
            )
            logger.info(f"\n✓ Training complete!")
            logger.info(f"  Model saved: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Train Donkeycar model with sim + real data'
    )
    parser.add_argument('--config',
                        help='Path to mycar config.py (optional)')
    parser.add_argument('--model', required=True,
                        choices=['linear', 'pilotnet', 'categorical', 'memory',
                                 'cnn_lstm', 'rnn', 'inferred'],
                        help='Model type (pilotnet recommended for sim2real)')
    parser.add_argument('--tub', required=True, nargs='+',
                        help='Tub paths (space-separated): sim_run_1 sim_run_2 real_run_1')
    parser.add_argument('--epochs', type=int,
                        help='Max epochs (default: config.MAX_EPOCHS)')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size (default: config.BATCH_SIZE)')
    parser.add_argument('--transfer',
                        help='Path to base model for transfer learning')

    args = parser.parse_args()

    # Crear entrenador
    trainer = Sim2RealTrainer(config_module=args.config)

    # Entrenar
    try:
        model_path = trainer.train_model(
            model_type=args.model,
            tub_paths=args.tub,
            transfer_from=args.transfer,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        logger.info(f"\nModel ready: {model_path}")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
