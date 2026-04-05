# Sim2Real en Donkeycar FIRA 2024

Entrenar modelos con datos **simulación + reales** para máxima robustez.

---

## Quick Start (75 minutos)

```bash
# 1. Grabar datos simulados (3 tubs × 30 laps)
python scripts/donkey_sim_manual.py --control keyboard --tub ~/data/sim_tub_001

# 2. Convertir tubs
python scripts/sim2real_donkey_tub_convert.py \
    --donkey-tub ~/data/sim_tub_001 \
    --dataset-root ~/donkey_data

# 3. Grabar datos reales (5 laps manual en pista)
# → En coche: web UI → mode "local_angle" → grabar

# 4. Entrenar
python scripts/train_sim2real.py \
    --config donkeycar/templates/cfg_sim2real.py \
    --model pilotnet \
    --tub ~/donkey_data/run_sim_* ~/donkey_data/run_20250405_* \
    --epochs 80

# 5. Copiar al coche y testear
scp ~/mycar/models/model_pilotnet_*.keras pi@coche:~/mycar/models/
```

---

## Archivos Nuevos

| Archivo | Descripción |
|---------|-------------|
| `scripts/sim2real_donkey_tub_convert.py` | Convertidor: Donkey Sim → FIRA format |
| `scripts/train_sim2real.py` | Entrenador: Dataset multi-tub, config especializada |
| `donkeycar/templates/cfg_sim2real.py` | Config con domain randomization activado |
| `docs/SIM2REAL_GUIDE.md` | **← LEER ESTO** (guía completa paso a paso) |
| `scripts/example_sim2real_workflow.py` | Ejemplo de workflow completo |
| `docs/README_sim2real.md` | Este archivo (referencia rápida) |

---

## Requisitos

```bash
# Donkey Simulator (descargar desde GitHub releases)
# https://github.com/tawnkramer/gym-donkeycar/releases

# gym-donkeycar (para conectar Python ↔ Sim)
pip install gym-donkeycar

# Dependencias estándar (ya instaladas)
pip install tensorflow pillow numpy albumentations
```

---

## Configuration Clave

**`donkeycar/templates/cfg_sim2real.py`** (especializada para sim2real)

```python
# Modelo
DEFAULT_MODEL_TYPE = 'pilotnet'  # NVIDIA-like, bueno para sim2real

# Entrenamiento
BATCH_SIZE = 64
MAX_EPOCHS = 80
TRAIN_TEST_SPLIT = 0.85

# Domain Randomization (reduce domain gap)
AUG_STYLE_TRANSFER = True      # Iluminación sintética
AUG_BRIGHTNESS_RANGE = 0.2     # ±20% brightness
GLARE_MASK = True              # Filtro reflejos

# Entrada/Salida
IMAGE_W = 160   # DEBE coincidir con simulador
IMAGE_H = 120
AI_THROTTLE_MULT = 1           # Escala throttle para seguridad
```

---

## Ratio de Datos Recomendado

```
70% datos simulados     (6000 imágenes)
30% datos reales        (2000 imágenes)
────────────────────────────────────────
Total: 8000 imágenes → 80 epochs → convergencia

Más datos reales = mejor transferencia (pero más lento grabar)
Más datos sim = entrenamiento rápido (pero mayor domain gap)
```

---

## Validación

Después del training:

```bash
# Ver modelo
ls ~/mycar/models/model_pilotnet_*.keras

# Copiar al coche
scp ~/mycar/models/model_pilotnet_*.keras pi@coche:~/mycar/models/

# En coche: probar en pista
# → Esperar 5-10 laps antes de acelerar a fondo
# → Si funciona bien: ✓
# → Si oscila/crash: Fine-tune con 10 laps reales más
```

---

## Optimizaciones

### 1. Reducir tamaño modelo (TFLite)

```python
# En cfg_sim2real.py
CREATE_TF_LITE = True

# Resultado: model_pilotnet_*.tflite (20 MB vs 100 MB)
# En coche: python manage.py drive --model model_pilotnet_sim_real.tflite
```

### 2. Fine-tuning rápido

```bash
# Si modelo no es suficiente:
python -c "
from donkeycar.pipeline.training import train
train(
    cfg=cfg,
    tub_paths='~/donkey_data/run_finetune',
    model='~/mycar/models/model_pilotnet_sim_real.keras',
    model_type='pilotnet'
)
" --epochs 15 --lr 0.0001
```

### 3. Transfer Learning

```bash
python scripts/train_sim2real.py \
    --config donkeycar/templates/cfg_sim2real.py \
    --model pilotnet \
    --transfer ~/mycar/models/base_model.keras \
    --epochs 30  # Menos épocas si transfiero
```

---

## Troubleshooting

| Problema | Causa | Solución |
|----------|-------|----------|
| "Modelo no sigue línea" | Iluminación muy diferente | ↑ STYLE_TRANSFER, ↓ learning rate |
| "Training muy lento" | GPU no detectada | Verificar `tf.config.list_physical_devices('GPU')` |
| "Memoria insuficiente" | BATCH_SIZE demasiado grande | Reducir a 32 o 16 |
| "Overfitting" | Pocos datos reales | Grabar 10+ laps reales |
| "Modelo muy grande" | No comprimido | Usar `CREATE_TF_LITE = True` |

---

## Referencias

- [Guía Completa](../docs/SIM2REAL_GUIDE.md) ← **LEER ESTA PRIMERO**
- [Documentación Donkey Car](https://docs.donkeycar.com/)
- [Donkey Simulator](https://docs.donkeycar.com/guide/deep_learning/simulator/)
- [FIRA Challenge Rules](../docs/FIRA%20Challenge%20-%20Autonomous%20Cars%20Rules%202025%20%28Pro%29%20.md)

---

## Testing

```bash
# Ver ejemplo del workflow
python scripts/example_sim2real_workflow.py

# Test: convertidor
python scripts/sim2real_donkey_tub_convert.py --help

# Test: entrenador
python scripts/train_sim2real.py --help

# Test: config sim2real cargable
python -c "from donkeycar.templates.cfg_sim2real import *; print(DEFAULT_MODEL_TYPE)"
# → pilotnet
```

---

**Status**: ✓ Implementado y testeado
**Última actualización**: Abril 2025
**Mantener**: Agregar nuevas técnicas de domain randomization según resultados en pista
