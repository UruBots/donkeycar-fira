# Guia de ejecucion: FIRA Safety Arbiter + Safety Signals

Este documento explica como correr y validar la implementacion de seguridad FIRA agregada en este repositorio.

## 1. Que incluye esta implementacion

La solucion agrega una capa de seguridad sobre los comandos del piloto automatico:

- Estimador de senales de carril y obstaculo desde camara.
- Arbiter de seguridad para fusionar piloto, carril y correccion de obstaculo.
- Failsafe por estado de detectores YOLO/TF.
- Suavizado temporal (EMA) de senales para reducir jitter.
- Telemetria opcional para analisis post-carrera.

Archivos principales:

- donkeycar/parts/fira_safety_signals.py
- donkeycar/parts/fira_safety_arbiter.py
- donkeycar/templates/complete.py
- donkeycar/templates/cfg_complete.py
- donkeycar/templates/cfg_simulator.py

## 2. Requisitos

- Entorno Python funcional para el proyecto.
- Dependencias instaladas del repo.
- Para pruebas de vision: OpenCV disponible (cv2).

## 3. Configuracion minima

La implementacion ya viene activada por defecto en los templates principales.

En configuracion de carro real (cfg_complete.py):

- FIRA_SAFETY_ARBITER = True
- FIRA_SAFETY_SIGNAL_ESTIMATOR = True
- FIRA_SAFETY_USE_LANE_GUIDANCE = True
- FIRA_SAFETY_USE_OBSTACLE_CORRECTION = True

En simulador (cfg_simulator.py), los mismos flags tambien vienen activados.

## 4. Como correr validaciones

Desde la raiz del repo:

```bash
/bin/python -m pytest donkeycar/tests/test_fira_template_inputs.py donkeycar/tests/test_fira_safety_signals.py donkeycar/tests/test_fira_safety_arbiter.py donkeycar/tests/test_fira_health_metrics.py donkeycar/tests/test_fira_engine_yolo_health.py donkeycar/tests/test_fira_engine_tf_health.py -q
```

Si quieres ejecutar solo lo nuevo de senales/arbiter:

```bash
/bin/python -m pytest donkeycar/tests/test_fira_safety_signals.py donkeycar/tests/test_fira_safety_arbiter.py -q
```

## 5. Como correr en tu proyecto de carro

Esta logica vive en el template complete. Para usarla en un proyecto de carro:

1. Asegura que tu proyecto use el flujo basado en template complete.
2. Verifica que en tu archivo de config esten activos los flags de seguridad.
3. Arranca el modo drive con tu comando habitual del proyecto.

Ejemplo tipico en un proyecto donkeycar:

```bash
python manage.py drive --myconfig=myconfig.py --model models/<tu_modelo>
```

Notas:

- Si no usas YOLO/TF de FIRA, el arbiter sigue funcionando con piloto + senales de vision.
- Si usas motores de deteccion FIRA, el failsafe toma su estado de salud para limitar throttle cuando corresponda.

## 6. Parametros recomendados para tuning en pista

Ajusta primero estos parametros:

- FIRA_SAFETY_CURVE_START
- FIRA_SAFETY_CURVE_FULL
- FIRA_SAFETY_CURVE_FACTOR_MIN
- FIRA_SAFETY_OBSTACLE_BLEND_MAX
- FIRA_SAFETY_OBSTACLE_THROTTLE_FACTOR_MIN
- FIRA_SAFETY_SIGNAL_SMOOTHING_ALPHA
- FIRA_SAFETY_SIGNAL_LANE_MIN_PIXEL_RATIO
- FIRA_SAFETY_SIGNAL_OBSTACLE_MIN_PIXEL_RATIO

Regla practica:

- Si oscila mucho: baja smoothing alpha o baja obstacle blend.
- Si reacciona tarde al cono: sube obstacle blend o sube severidad efectiva (min/full pixel ratio).
- Si se pasa de velocidad en curvas: baja curve factor min.

## 7. Telemetria de seguridad (opcional)

Para guardar metricas de seguridad en tub/MQTT, habilita:

- FIRA_HEALTH_TELEMETRY = True

Con eso se registran, entre otras:

- fira/safety/failsafe_active
- fira/safety/lane_weight
- fira/safety/obstacle_weight
- fira/safety/curve_factor
- fira/safety/speed_limit_factor

## 8. Checklist rapido antes de competir

- Tests FIRA en verde.
- Camara calibrada y sin zonas ciegas de ROI.
- HSV de carril y obstaculo ajustados a iluminacion real.
- Throttle caps de failsafe validados en pista.
- Prueba en pista externa con conos y en pista interna urbana.

## 9. Entrenamiento con robustez visual

Ya estan disponibles los overrides explicitos para entrenamiento:

- `--mask-glare` / `--no-mask-glare`
- `--style-transfer` / `--no-style-transfer`
- `--style-transfer-preset=<preset>`
- `--style-transfer-blend=<blend>`

Ejemplo:

```bash
donkey train --tub data --model models/mypilot.h5 --type linear --mask-glare --style-transfer --style-transfer-preset sepia --style-transfer-blend 0.35
```

En la UI de entrenamiento, estos mismos flags se pueden ajustar desde el editor de config antes de pulsar Train:

- `GLARE_MASK = True`
- `AUG_STYLE_TRANSFER = True`
- `AUG_STYLE_TRANSFER_PRESET = 'sepia'` o `random`
- `AUG_STYLE_TRANSFER_BLEND = 0.35`

Tambien hay toggles directos en la pantalla de entrenamiento para activar o desactivar glare mask y style transfer sin editar el config manualmente.

Durante el entrenamiento, el modelo guarda una metadata lateral en `<model_path>.metadata.json`. En carga, esa metadata se re-aplica para activar `GLARE_MASK` por modelo sin depender solo del config global.

## 10. Estado actual del plan

Queda pendiente solo la validacion final en hardware objetivo y, si quieres llevarlo al extremo, una pasada de tuning fino sobre los umbrales HSV/ROI para la pista real.

Lo que ya esta cerrado:

- Safety arbiter y safety signals.
- Failsafe por salud de detectores.
- Glare mask en entrenamiento e inferencia.
- Style transfer sintético en entrenamiento.
- Overrides de entrenamiento por CLI/API.
- Metadata por modelo para glare.
- Pruebas de regresion de todo lo anterior.
