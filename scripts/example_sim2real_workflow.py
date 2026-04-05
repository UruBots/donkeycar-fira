#!/usr/bin/env python3
"""
Ejemplo: Workflow completo sim2real

Este script muestra cómo: 
1. Convertir tubs del simulador
2. Entrenar con datos mezclados
3. Generar modelo optimizado

Uso:
    python example_sim2real_workflow.py
"""

import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Agregar proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def step_1_convert_sim_tubs():
    """Paso 1: Convertir tubs del simulador"""
    print("\n" + "="*70)
    print("STEP 1: Convertir Tubs del Donkey Simulator → Formato FIRA")
    print("="*70)
    
    # Rutas de ejemplo
    sim_tubs = [
        "~/data/sim_tub_001",
        "~/data/sim_tub_002",
        "~/data/sim_tub_003",
    ]
    output_dir = "~/donkey_data"
    
    logger.info(f"""
Convertir tubs simulados a formato compatible con FIRA training.
Uso:
    python scripts/sim2real_donkey_tub_convert.py \\
        --donkey-tub {sim_tubs[0]} \\
        --dataset-root {output_dir} \\
        --img-w 160 --img-h 120 \\
        --prefix run_sim_
        
Repetir para cada tub:
""")
    for tub in sim_tubs:
        cmd = f"""    python scripts/sim2real_donkey_tub_convert.py \\
        --donkey-tub {tub} \\
        --dataset-root {output_dir} \\
        --img-w 160 --img-h 120 --prefix run_sim_"""
        print(cmd)
    
    print(f"""
Resultado esperado:
    {output_dir}/run_sim_20250405_143022/
    {output_dir}/run_sim_20250405_145301/
    {output_dir}/run_sim_20250405_151045/
    └── Cada uno con manifest.json + catalog + images/
""")


def step_2_prepare_real_data():
    """Paso 2: Preparar datos reales"""
    print("\n" + "="*70)
    print("STEP 2: Grabar Datos Reales (Pista FIRA)")
    print("="*70)
    
    logger.info("""
En el coche FIRA:
    1. Abrir web UI: http://coche_ip:8887
    2. Modo "local_angle" (AI mode with recording)
    3. Dar 5 laps en manual (teclado/joystick)
    4. Datos guardados en ~/mycar/data/run_YYYYMMDD_*/
    
En PC de entrenamiento:
    scp -r pi@coche:~/mycar/data/run_* ~/donkey_data/
    
Resultado esperado:
    ~/donkey_data/run_20250405_152000/
    └── manifest.json + catalog + images/
""")


def step_3_train_sim2real():
    """Paso 3: Entrenar modelo mixto"""
    print("\n" + "="*70)
    print("STEP 3: Entrenar Modelo con Datos Sim + Real")
    print("="*70)
    
    logger.info("""
Opción A: Script personalizado (recomendado)
    
    python scripts/train_sim2real.py \\
        --config donkeycar/templates/cfg_sim2real.py \\
        --model pilotnet \\
        --tub ~/donkey_data/run_sim_* ~/donkey_data/run_20250405_* \\
        --epochs 80 \\
        --batch-size 64

Opción B: API directa (si hay problemas)

    python -c "
import sys
sys.path.insert(0, '.')
from donkeycar.config import Config
from donkeycar.pipeline.training import train

# Cargar config sim2real
from donkeycar.templates.cfg_sim2real import *
cfg = Config()

# Paths de tubs (comma-separated)
tub_paths = '~/donkey_data/run_sim_*,~/donkey_data/run_20250405_*'

# Entrenar
model_path = train(
    cfg=cfg,
    tub_paths=tub_paths,
    model_type='pilotnet'
)
print(f'Modelo entrenado: {model_path}')
"

Salida esperada:
    ~/mycar/models/model_pilotnet_<timestamp>.keras
    ~/mycar/models/model_pilotnet_<timestamp>.tflite  (optimizado)
""")


def step_4_deploy_and_test():
    """Paso 4: Desplegar y testear"""
    print("\n" + "="*70)
    print("STEP 4: Desplegar en Coche y Testear")
    print("="*70)
    
    logger.info("""
Copiar modelo al coche:
    
    scp ~/mycar/models/model_pilotnet_*.keras pi@coche:~/mycar/models/

En el coche:
    1. Recargar página web
    2. Seleccionar modelo en dropdown
    3. Cambiar a "local_angle" mode
    4. Probar en pista (empezar LENTO)
    
Si todo va bien:
    ✓ Modelo sigue líneas
    ✓ Velocidad controlada
    ✓ Sin crashes
    
Si no:
    → Fine-tune con 10 laps reales más
    → Reducir AI_THROTTLE_MULT a 0.5 para testing
    → Revisar iluminación (¿muy diferente a sim?)
""")


def step_5_optional_finetune():
    """Paso 5: Fine-tuning opcional"""
    print("\n" + "="*70)
    print("STEP 5: Fine-tuning (Opcional)")
    print("="*70)
    
    logger.info("""
Si rendimiento no es óptimo, hacer fine-tuning:
    
    1. Grabar 10 laps reales más
    2. Guardar en ~/donkey_data/run_finetune/
    
    python -c "
from donkeycar.config import Config
from donkeycar.pipeline.training import train

cfg = Config()
cfg.MAX_EPOCHS = 15      # Pocas épocas
cfg.LEARNING_RATE = 0.0001  # Learning rate bajo

train(
    cfg=cfg,
    tub_paths='~/donkey_data/run_finetune',
    model='~/mycar/models/model_pilotnet_sim_real.keras',
    model_type='pilotnet'
)
"

Resultado:
    Modelo mejorado con características específicas de pista real
""")


def main():
    """Ejecutar workflow"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    SIM2REAL WORKFLOW - FIRA 2024                      ║
║                                                                        ║
║  Entrenar modelo Donkeycar mezclando datos simulados + datos reales  ║
╚════════════════════════════════════════════════════════════════════════╝
""")
    
    try:
        step_1_convert_sim_tubs()
        step_2_prepare_real_data()
        step_3_train_sim2real()
        step_4_deploy_and_test()
        step_5_optional_finetune()
        
        print("\n" + "="*70)
        print("✓ Workflow complete!")
        print("="*70)
        print("""
Resumen:
  1. Grabar 3 tubs simulados (90 laps, ~10 min)
  2. Convertir a formato FIRA (~5 min)
  3. Grabar 5 laps reales (~15 min)
  4. Entrenar modelo pilotnet (80 epochs, ~30 min)
  5. Desplegar en coche y testear
  6. Fine-tune opcional (si necesario)
  
TOTAL: ~75 minutos desde cero a coche funcionando

Documentación completa: docs/SIM2REAL_GUIDE.md
""")
        return 0
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
