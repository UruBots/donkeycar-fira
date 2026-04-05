# Sim2Real Entrenamiento: Mezcla Simulación + Datos Reales

Estrategia práctica para entrenar un modelo Donkeycar-FIRA con **datos sintéticos (Donkey Simulator)** y **datos reales (pista FIRA)**, logrando convergencia robusta y transferencia a cancha real.

---

## 📋 Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Requisitos](#requisitos)
3. [Fase 1: Grabar Datos Simulados](#fase-1-grabar-datos-simulados)
4. [Fase 2: Convertir Tubs Simulados](#fase-2-convertir-tubs-simulados)
5. [Fase 3: Grabar Datos Reales](#fase-3-grabar-datos-reales)
6. [Fase 4: Entrenar Modelo Mixto](#fase-4-entrenar-modelo-mixto)
7. [Fase 5: Fine-tuning en Real](#fase-5-fine-tuning-en-real-opcional)
8. [Troubleshooting](#troubleshooting)

---

## Visión General

```
Simulación (70% datos)              Pista Real (30% datos)
      ↓                                   ↓
   Donkey Sim                         Conducción Manual
      ↓                                   ↓
  30 laps × 128px                   5 laps × 160px
      ↓                                   ↓
  Convertir → FIRA format           Grabación automática
      ↓                                   ↓
  Datos sintéticos                  Datos reales
      └────────────────┬──────────────────┘
                       ↓
              Entrenar con mix
              (80 epochs, pilotnet)
                       ↓
              Modelo transferible
                       ↓
         Fine-tune en real (opcional)
                       ↓
           Usar en pista FIRA
```

### ¿Por qué funciona?

- **Domain Randomization**: El simulador añade variación en iluminación, texturas
- **Feature-based transfer**: El modelo aprende características robustas, no texturas
- **Augmentación en training**: Brightness, blur, style transfer emula variación real
- **Fine-tuning ligero**: Pocas épocas en real adapta al dominio específico

---

## Requisitos

### Software

```bash
# 1. Donkey Car (este repo)
cd /home/utec/Desarrollo/donkeycar-fira-2024

# 2. Donkey Simulator (descargar desde)
# https://github.com/tawnkramer/gym-donkeycar/releases
# Ejemplo Linux:
wget https://github.com/tawnkramer/gym-donkeycar/releases/download/v23.0/DonkeySimLinux.zip
unzip DonkeySimLinux.zip

# 3. gym-donkeycar (para conectar Python ↔ Sim)
pip install gym-donkeycar

# 4. Dependencias (ya instaladas si setup completo)
pip install tensorflow>=2.4 numpy pillow albumentations
```

### Hardware

- **Sim2real**: PC con GPU (entrenamiento más rápido)
- **Grabación real**: Coche FIRA + pista 50cm ancho

### Pista en Simulador

El mapa simulado debe tener:
- **Ancho**: 50 cm (coincidir con pista FIRA)
- **Líneas**: Blancas/negras o AprilTags
- **Iluminación**: Variable (ajustar después)
- **Cámaracámara**: 160×120 (configurable en gym-donkeycar)

---

## Fase 1: Grabar Datos Simulados

### Paso 1: Arrancar el Simulador

```bash
cd ~/Downloads/DonkeySimLinux
./donkey_sim.x86_64 &
```

El simulador abrirá un menú. Seleccionar:
- **Escena**: `generated_track` (o tu pista personalizada)
- **Modo**: Recording

### Paso 2: Grabar Tub (modo manual)

Opción A: **Conducción con teclado** (W/A/S/D + Q=salir)

```bash
cd /home/utec/Desarrollo/donkeycar-fira-2024

python scripts/donkey_sim_manual.py \
    --control keyboard \
    --sim-env donkey-generated-track-v0 \
    --tub ~/data/sim_tub_001
```

- Pulsa **W** para acelerar, **A/D** para girar
- Toma ~30 laps (≈2000 fotogramas) = ~80 segundos
- Variar iluminación/posición en simulador entre runs

Opción B: **Conducción automática** (con modelo base)

```bash
python scripts/donkey_sim_drive.py \
    --model path/to/model.keras \
    --sim-env donkey-generated-track-v0 \
    --tub ~/data/sim_tub_001
```

### Paso 3: Verificar Tub Grabado

```bash
ls -lh ~/data/sim_tub_001/
# Debe tener:
# - manifest.json
# - catalog_0.catalog
# - images/ (2000+ .jpg)
```

**Repetir 2-3 veces con variaciones:**
- Cambiar FOV en simulador (Field of View 90-150°)
- Variar brightness (menu → Settings)
- Grabar en "pista alternativa" si está disponible

**Resultado esperado**: 3 tubs simulados
- `sim_tub_001/` → 2000 registros
- `sim_tub_002/` → 2000 registros  
- `sim_tub_003/` → 2000 registros
- **Total**: ~6000 registros simulados

---

## Fase 2: Convertir Tubs Simulados

El Donkey Simulator guarda imágenes en su propio formato. Convertir a formato FIRA compatible.

### Paso 1: Usar el convertidor

```bash
cd /home/utec/Desarrollo/donkeycar-fira-2024

python scripts/sim2real_donkey_tub_convert.py \
    --donkey-tub ~/data/sim_tub_001 \
    --dataset-root ~/donkey_data \
    --img-w 160 \
    --img-h 120 \
    --prefix run_sim_
```

**Parámetros:**
- `--donkey-tub`: Ruta del tub del simulador
- `--dataset-root`: Folder raíz para guardar runs convertidos
- `--img-w 160 --img-h 120`: Resolución FIRA (IMPORTANTE: debe coincidir)
- `--prefix run_sim_`: Prefijo para identificar runs sintéticos

**Salida:**
```
✓ Conversion complete!
  Processed: 2000 images
  Skipped: 0 records
  Output: /home/utec/data/run_sim_20250405_143022
  Use in training: --tub-paths /home/utec/data/run_sim_20250405_143022
```

### Paso 2: Repetir para otros tubs

```bash
python scripts/sim2real_donkey_tub_convert.py \
    --donkey-tub ~/data/sim_tub_002 \
    --dataset-root ~/donkey_data \
    --img-w 160 --img-h 120 --prefix run_sim_

python scripts/sim2real_donkey_tub_convert.py \
    --donkey-tub ~/data/sim_tub_003 \
    --dataset-root ~/donkey_data \
    --img-w 160 --img-h 120 --prefix run_sim_
```

### Resultado

```bash
ls ~/donkey_data/
# run_sim_20250405_143022/
# run_sim_20250405_145301/
# run_sim_20250405_151045/
```

---

## Fase 3: Grabar Datos Reales

### Requisitos

1. Coche FIRA configurado y funcionando
2. Pista de 50cm ancho (línea blanca o AprilTags)
3. Iluminación similar a simulador

### Paso 1: Inicializar el Coche

En el coche (Raspberry Pi o local):

```bash
ssh pi@donkey_car_ip  # Si es remoto
cd ~/mycar
python manage.py drive
```

- Abrir interfaz web: `http://donkey_car_ip:8887`
- Seleccionar **MODE: local_angle** (AI mode con recording)
- Verificar que se ve la imagen de cámara

### Paso 2: Grabar Manual para Recolección Base

- **Control**: Teclado o joystick (seleccionar en UI)
- **Modo**: `data` (recordar datos)
- **Dar 5 laps** en la pista en modo manual

El sistema grabará automáticamente `run_YYYYMMDD_HHMMSS/` en `~/mycar/data/`

### Paso 3: Transferir Données a PC de Entrenamiento

```bash
scp -r pi@donkey_car_ip:~/mycar/data/run_* ~/donkey_data/
```

O si está en el mismo PC:
```bash
# Las grabaciones ya están en ~/donkey_data/
```

**Verificar estructura:**
```bash
ls ~/donkey_data/run_20250405_*/
# catalog_0.catalog
# manifest.json
# images/
```

---

## Fase 4: Entrenar Modelo Mixto

### Paso 1: Preparar Rutas de Tubs

```bash
# Variables con rutas
SIM_TUB1=~/donkey_data/run_sim_20250405_143022
SIM_TUB2=~/donkey_data/run_sim_20250405_145301
SIM_TUB3=~/donkey_data/run_sim_20250405_151045
REAL_TUB=~/donkey_data/run_20250405_152000

# Verificar que existen
for tub in $SIM_TUB1 $SIM_TUB2 $SIM_TUB3 $REAL_TUB; do
    [ -d "$tub" ] && echo "✓ $tub" || echo "✗ MISSING: $tub"
done
```

### Paso 2: Entrenar con Script Personalizado

```bash
cd /home/utec/Desarrollo/donkeycar-fira-2024

python scripts/train_sim2real.py \
    --config donkeycar/templates/cfg_sim2real.py \
    --model pilotnet \
    --tub ~/donkey_data/run_sim_* ~/donkey_data/run_20250405_* \
    --epochs 80 \
    --batch-size 64
```

**O sin script (usando API directa):**

```bash
python -c "
import sys
sys.path.insert(0, '.')
from donkeycar.config import Config
from donkeycar.pipeline.training import train

cfg = Config()  # Carga cfg_complete.py por defecto
# O: from donkeycar.templates.cfg_sim2real import *

tub_paths = '~/donkey_data/run_sim_*,~/donkey_data/run_20250405_*'

model_path = train(
    cfg=cfg,
    tub_paths=tub_paths,
    model_type='pilotnet'
)
print(f'Model: {model_path}')
"
```

### Paso 3: Monitorear Entrenamiento

```
Epoch 1/80
2500/2500 [==============================] - 412s - loss: 0.0245 - val_loss: 0.0218
Epoch 2/80
2500/2500 [==============================] - 408s - loss: 0.0156 - val_loss: 0.0144
...
Epoch 80/80
2500/2500 [==============================] - 405s - loss: 0.0032 - val_loss: 0.0031
Save best models...
```

- **Loss debe descender** (primeras épocas rápido, luego lento)
- **Val loss similar a loss** (si diverge mucho → overfitting)
- Parar si no hay progreso (early stopping)

### Paso 4: Modelo Entrenado

```bash
ls ~/mycar/models/
# model_pilotnet_20250405_165000.keras  ← Modelo final
# model_pilotnet_20250405_165000.h5    ← Alternativo
# model_pilotnet_20250405_165000.tflite ← Optimizado
```

**Copiar al coche:**
```bash
scp ~/mycar/models/model_pilotnet_*.keras pi@donkey_car_ip:~/mycar/models/
```

---

## Fase 5: Fine-tuning en Real (Opcional)

Si el modelo inicial no es suficientemente bueno en pista real, fine-tune con 5-10 laps reales más.

### Paso 1: Grabar Datos Reales Adicionales

```bash
# En la UI del coche: conducir más laps en modo manual
# Genera: ~/mycar/data/run_20250405_finetune/
```

### Paso 2: Transferir

```bash
scp -r pi@donkey_car_ip:~/mycar/data/run_20250405_finetune ~/donkey_data/
```

### Paso 3: Fine-tune

```bash
python -c "
from donkeycar.config import Config
from donkeycar.pipeline.training import train

cfg = Config()
cfg.MAX_EPOCHS = 10  # Pocas épocas
cfg.LEARNING_RATE = 0.0001  # Learning rate bajo

model_path = train(
    cfg=cfg,
    tub_paths='~/donkey_data/run_20250405_finetune',
    model='~/mycar/models/model_pilotnet_sim_real.keras',  # Base model
    model_type='pilotnet'
)
print(f'Fine-tuned model: {model_path}')
"
```

---

## Optimizaciones Avanzadas

### 1. Domain Randomization en Simulador

En `cfg_sim2real.py`:
```python
AUG_STYLE_TRANSFER = True      # Iluminación sintética
AUG_BRIGHTNESS_RANGE = 0.2     # ±20% variación brightness
AUG_BLUR_RANGE = (1, 3)        # Blur simulado
GLARE_MASK = True              # Filtro reflejos
```

### 2. Usar Modelos Ligeros para Inferencia

Después del training, generar versiones optimizadas:

```python
# Automático si CREATE_TF_LITE = True en cfg
# Resultado: model_pilotnet_*.tflite

# En el coche usar:
python manage.py drive --model model_pilotnet_sim_real.tflite
```

### 3. Confidence-based Fallback

Usar modelo con score de confianza:
```python
DEFAULT_MODEL_TYPE = 'confidence'  # Retorna conf score

# Si conf < CONFIDENCE_THRESHOLD → neutral
CONFIDENCE_THRESHOLD = 0.5
```

---

## Troubleshooting

### ❌ "Modelo terrible en pista real"

**Causas:**
- Iluminación muy diferente entre sim y real
- Resolución no coincide (160×120 sim ≠ real)
- Pista real tiene cambios que sim no tiene

**Soluciones:**
1. Aumentar datos reales (grabar 10+ laps, no 5)
2. Fine-tune 20 épocas (no 10)
3. Activar GLARE_MASK y STYLE_TRANSFER
4. Reducir AI_THROTTLE_MULT a 0.5 para testing

### ❌ "Training muy lento"

**Causas:**
- GPU no se usa (TensorFlow no detectó CUDA)
- BATCH_SIZE demasiado grande
- Disco lento

**Soluciones:**
```bash
# Verificar GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Reducir batch
cfg.BATCH_SIZE = 32  # de 64

# Reducir datos: muestreo aleatorio 50%
```

### ❌ "Error: Missing images en tub"

**Verificar conversor:**
```bash
python scripts/sim2real_donkey_tub_convert.py \
    --donkey-tub ~/data/sim_tub_001 \
    --dataset-root ~/test
# Revisar logs: ¿Skipped cuántos?
```

### ❌ "Modelo muy grande (> 100 MB)"

Usar versión TFLite:
```python
CREATE_TF_LITE = True  # En cfg_sim2real.py
# Resultado: model.tflite (~20 MB)
```

---

## Resumen: Pasos Clave

| Paso | Acción | Tiempo | Archivos |
|------|--------|--------|----------|
| 1 | Grabar 3 tubs simulados (90 laps) | 10 min | `sim_tub_001,002,003/` |
| 2 | Convertir tubs → formato FIRA | 5 min | `run_sim_*/manifest.json` |
| 3 | Grabar 5 laps reales (manual) | 15 min | `run_20250405_.../` |
| 4 | Entrenar modelo (80 epochs) | 30 min | `model_pilotnet.keras` |
| 5 | Probar en pista real | 10 min | Iteración |
| 6 | Fine-tune (opcional) | 5 min | `model_pilotnet_tuned.keras` |
| **TOTAL** | | **75 min** | |

---

## Referencias

- [Donkey Car Docs – Training](https://docs.donkeycar.com/guide/train_autopilot/)
- [Donkey Simulator – Official](https://docs.donkeycar.com/guide/deep_learning/simulator/)
- [gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar)
- [FIRA Challenge Rules 2025](../../docs/FIRA%20Challenge%20-%20Autonomous%20Cars%20Rules%202025%20%28Pro%29%20.md)

---

**Última actualización**: Abril 2025 | **Status**: Probado en FIRA 2024 | **Soporte**: equipo Donkeycar-FIRA
