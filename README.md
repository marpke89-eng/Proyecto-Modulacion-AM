# Proyecto-Modulacion-AM
Implementación y análisis de un sistema de modulación en amplitud (AM) en Python
# Simulación y Análisis de Modulación en Amplitud (AM)

Este proyecto implementa un sistema de modulación en amplitud (AM) en Python para analizar su comportamiento en los dominios del tiempo y la frecuencia, así como su desempeño frente a ruido, distorsión y atenuación.

## Requisitos
* Python 3.8+
* NumPy (`pip install numpy`)
* Matplotlib (`pip install matplotlib`)

## Estructura del Análisis
1. **Generación de señales:** Mensaje de baja frecuencia (100 Hz) y portadora (1500 Hz).
2. **Modulación AM:** Evaluación con índice óptimo ($m = 0.8$) y sobremodulación ($m = 1.5$).
3. **Transformada Rápida de Fourier (FFT):** Identificación de portadora y bandas laterales (USB/LSB).
4. **Degradación de canal:** Simulación de ruido blanco gaussiano (AWGN) y atenuación del 70%.

## Ejecución
```bash
python modulacion_am.py
```
