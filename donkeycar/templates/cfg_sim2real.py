#
# Configuración especializada para entrenamiento SIM2REAL
# Donkeycar FIRA 2024
#
# Uso:
#   python manage.py train --config cfg_sim2real.py
#
# O con el script personalizado:
#   python train_sim2real.py --config cfg_sim2real.py --model pilotnet
#

# Importar baseline
from donkeycar.templates.cfg_complete import *

print("=" * 60)
print("CONFIG: SIM2REAL Donkeycar FIRA 2024")
print("=" * 60)

# ============================================================================
# ENTRENAMIENTO: Parámetros para mezclar datos sim + real
# ============================================================================

# Usar modelo PilotNet (recomendado para sim2real)
DEFAULT_MODEL_TYPE = 'pilotnet'

# Batch más pequeño si memoria GPU limitada
BATCH_SIZE = 64  # Reducido de 128 para mezcla sim + real

# Entrenamiento más largo para convergencia
MAX_EPOCHS = 80  # Aumentado para stabilizar con datos mezclados

# Train/test: más datos de validación dada pista simplificada
TRAIN_TEST_SPLIT = 0.85

# Early stopping agresivo (no improve = parar)
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 8
MIN_DELTA = 0.0005

# Learning rate más bajo para fine-tuning mixto
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.0

print(f"Model: {DEFAULT_MODEL_TYPE}")
print(f"Training: {MAX_EPOCHS} epochs, batch {BATCH_SIZE}, train/val {TRAIN_TEST_SPLIT}/{1-TRAIN_TEST_SPLIT}")

# ============================================================================
# AUGMENTACIONES: Domain Randomization para sim2real
# ============================================================================
# Las augmentaciones solo se aplican en TRAINING, no en inferencia.
# Esto reduce el domain gap entre simulación y realidad.

# 1. STYLE TRANSFER: Sintetiza variaciones de iluminación/textura
AUG_STYLE_TRANSFER = True
AUG_STYLE_TRANSFER_PRESET = 'random'  # random, sunset, night, etc.
AUG_STYLE_TRANSFER_BLEND = 0.35       # Mezcla con imagen original

# 2. AUGMENTACIONES ESTÁNDAR
AUGMENTATIONS = [
    'STYLE_TRANSFER',  # Domain randomization: iluminación sintética
    'BRIGHT_AND_BLUR', # Variación brightness/blur
]

# Rango brightness: [-0.2, 0.2] → simula cambios iluminación
AUG_BRIGHTNESS_RANGE = 0.2

# Blur simulado: kernels 1-3 píxeles
AUG_BLUR_RANGE = (1, 3)

# ============================================================================
# TRANSFORMACIONES: Pre-procesado consistente (sim + real)
# ============================================================================
# Las transformaciones se aplican SIEMPRE (train + inferencia)

# 1. GLARE MASK: Cubre zonas de sobreexposición (reflejos)
# Detecta píxeles muy blancos y reemplaza con promedio
GLARE_MASK = True
GLARE_MASK_SAT_LOW = 80        # Saturation baja (zonas blancas)
GLARE_MASK_VAL_HIGH = 240      # Value alta (zonas brillantes)
GLARE_MASK_FILL_WITH = 'mean'  # Rellenar con promedio de región
GLARE_MASK_MORPH_KERNEL = 3
GLARE_MASK_MORPH_ITERATIONS = 1

TRANSFORMATIONS = ['GLARE_MASK']

# 2. ROI CROP (post-transformaciones)
# Para FIRA: NO recortar (usar imagen completa 160x120)
ROI_CROP_TOP = 0
ROI_CROP_BOTTOM = 0
ROI_CROP_RIGHT = 0
ROI_CROP_LEFT = 0

POST_TRANSFORMATIONS = []

# ============================================================================
# MODELO
# ============================================================================

# PilotNet: similar a NVIDIA autopilot, bueno para sim2real
# - Entrada: 160×120×3 imagen RGB normalizada [-0.5, 0.5]
# - Salida: [angle, throttle] ambos en [-1, 1]
DEFAULT_MODEL_TYPE = 'pilotnet'

# Sequence length para modelos temporales (RNN/LSTM)
SEQUENCE_LENGTH = 3

# Transfer learning: congelar primeras capas si usas modelo pre-entrenado
FREEZE_LAYERS = False  # Cambiar a True si haces transfer learning
NUM_LAST_LAYERS_TO_TRAIN = 7

# ============================================================================
# ENTRADA DE DATOS (asegurar consistencia sim/real)
# ============================================================================

# Resolución: Donkey Simulator DEBE usar la misma resolución
IMAGE_W = 160   # Ancho (DEBE coincidir con sim)
IMAGE_H = 120   # Alto (DEBE coincidir con sim)
IMAGE_DEPTH = 3 # RGB

# Normalización: [-0.5, 0.5] aplicada automáticamente
# (uint8 [0,255] → float32 [-0.5, 0.5])

# ============================================================================
# SALIDA (Control)
# ============================================================================

# PilotNet: salida [-1, 1] para steering y throttle
# El servo/ESC mapea esto a PWM

# Multiplicador throttle IA para FIRA (seguridad)
AI_THROTTLE_MULT = 1  # Escala salida de IA (p.ej. 0.8 = 80%)

# Control type: PWM estándar para RC car FIRA
DRIVE_TRAIN_TYPE = "PWM_STEERING_THROTTLE"

# ============================================================================
# CONVERSIÓN MODELOS
# ============================================================================

# Generar automáticamente durante training
CREATE_TF_LITE = True   # Exportar .tflite para optimización
CREATE_TENSOR_RT = False # TensorRT si tenés GPU NVIDIA

# ============================================================================
# VALIDACIÓN / TESTING
# ============================================================================

# Verbose training
VERBOSE_TRAIN = True

# Cache: ARRAY para velocidad (requiere RAM)
CACHE_POLICY = 'ARRAY'  # 'NOCACHE' si memoria limitada

print("\n" + "="*60)
print("Domain Randomization Settings:")
print(f"  Style transfer:     {AUG_STYLE_TRANSFER} (iluminación/textura sintética)")
print(f"  Brightness range:   ±{AUG_BRIGHTNESS_RANGE*100:.0f}%")
print(f"  Glare mask:         {GLARE_MASK} (filtro reflejos)")
print("="*60)
