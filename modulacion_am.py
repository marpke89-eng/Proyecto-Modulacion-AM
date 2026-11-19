import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. DEFINICIÓN DE PARÁMETROS Y SEÑALES DE ENTRADA
# ==============================================================================
fs = 10000  # Frecuencia de muestreo (Hz) - debe ser >> 2 * fc (Teorema de Nyquist)
t = np.arange(0, 0.05, 1 / fs)  # Vector de tiempo de 50 ms

# Parámetros de la señal mensaje (información de baja frecuencia)
fm = 100  # Frecuencia del mensaje (Hz)
Am = 1.0  # Amplitud del mensaje (V)
mensaje = Am * np.cos(2 * np.pi * fm * t)

# Parámetros de la señal portadora (alta frecuencia)
fc = 1500  # Frecuencia de la portadora (Hz)
Ac = 1.0  # Amplitud de la portadora (V)
portadora = Ac * np.cos(2 * np.pi * fc * t)

# ==============================================================================
# 2. IMPLEMENTACIÓN DE LA MODULACIÓN EN AMPLITUD (AM)
# ==============================================================================
# Fórmula AM estándar: s(t) = Ac * [1 + m * cos(2*pi*fm*t)] * cos(2*pi*fc*t)
# Índice de modulación (m = Am / Ac). Si m <= 1 no hay distorsión de envolvente.
m_normal = 0.8  # Modulación óptima (80%)
s_am = Ac * (1 + m_normal * (mensaje / Am)) * portadora

# Escenario de distorsión por sobremodulación (m > 1)
m_sobre = 1.5  # Sobremodulación (150% -> inversión de fase y distorsión)
s_am_distorsion = Ac * (1 + m_sobre * (mensaje / Am)) * portadora

# ==============================================================================
# 3. ESCENARIOS DE CANAL: RUIDO Y ATENUACIÓN
# ==============================================================================
# Escenario A: Atenuación (pérdida de potencia por distancia/medio)
factor_atenuacion = 0.3  # La señal pierde el 70% de su amplitud
s_am_atenuada = s_am * factor_atenuacion

# Escenario B: Ruido Blanco Gaussiano Aditivo (AWGN)
np.random.seed(42)  # Semilla para reproducibilidad
potencia_ruido = 0.35
ruido = np.random.normal(0, potencia_ruido, len(t))
s_am_ruido = s_am + ruido


# ==============================================================================
# 4. FUNCIÓN PARA ANÁLISIS EN EL DOMINIO DE LA FRECUENCIA (FFT)
# ==============================================================================
def calcular_espectro(senal, fs):
  N = len(senal)
  fft_vals = np.fft.fft(senal)
  # Tomamos solo la mitad positiva del espectro y normalizamos
  frecuencias = np.fft.fftfreq(N, 1 / fs)[: N // 2]
  magnitud = (2.0 / N) * np.abs(fft_vals[: N // 2])
  return frecuencias, magnitud


freq_am, mag_am = calcular_espectro(s_am, fs)
freq_ruido, mag_ruido = calcular_espectro(s_am_ruido, fs)

# ==============================================================================
# 5. VISUALIZACIÓN DE RESULTADOS (TIEMPO Y FRECUENCIA)
# ==============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle(
    'Análisis de Sistema de Modulación en Amplitud (AM)',
    fontsize=16,
    fontweight='bold',
)

# Gráfico 1: Señal Mensaje vs Portadora (Tiempo)
axs[0, 0].plot(
    t[:300] * 1000,
    mensaje[:300],
    label='Mensaje m(t) [100 Hz]',
    color='blue',
    linewidth=2,
)
axs[0, 0].plot(
    t[:300] * 1000,
    portadora[:300],
    label='Portadora c(t) [1500 Hz]',
    color='orange',
    alpha=0.5,
)
axs[0, 0].set_title('1. Señal Mensaje y Portadora (Tiempo)')
axs[0, 0].set_xlabel('Tiempo (ms)')
axs[0, 0].set_ylabel('Amplitud (V)')
axs[0, 0].legend(loc='upper right')

# Gráfico 2: Señal Modulada AM Ideal (Tiempo)
axs[0, 1].plot(
    t[:300] * 1000, s_am[:300], color='green', label='Señal AM (m = 0.8)'
)
axs[0, 1].plot(
    t[:300] * 1000,
    Ac * (1 + m_normal * mensaje[:300]),
    'r--',
    label='Envolvente',
)
axs[0, 1].set_title('2. Señal Modulada AM Óptima (Tiempo)')
axs[0, 1].set_xlabel('Tiempo (ms)')
axs[0, 1].set_ylabel('Amplitud (V)')
axs[0, 1].legend(loc='upper right')

# Gráfico 3: Espectro de Frecuencia AM Ideal (FFT)
axs[1, 0].stem(freq_am, mag_am, linefmt='g-', markerfmt='go', basefmt=' ')
axs[1, 0].set_xlim(1000, 2000)  # Zoom alrededor de la portadora (1500 Hz)
axs[1, 0].set_title('3. Espectro de Frecuencia AM (FFT)')
axs[1, 0].set_xlabel('Frecuencia (Hz)')
axs[1, 0].set_ylabel('Magnitud (V)')
axs[1, 0].annotate(
    'Portadora (1500 Hz)',
    xy=(1500, 1.0),
    xytext=(1530, 0.9),
    arrowprops=dict(arrowstyle='->'),
)
axs[1, 0].annotate(
    'LSB (1400 Hz)',
    xy=(1400, 0.4),
    xytext=(1220, 0.45),
    arrowprops=dict(arrowstyle='->'),
)
axs[1, 0].annotate(
    'USB (1600 Hz)',
    xy=(1600, 0.4),
    xytext=(1650, 0.45),
    arrowprops=dict(arrowstyle='->'),
)

# Gráfico 4: Impacto del Ruido (Tiempo)
axs[1, 1].plot(
    t[:300] * 1000,
    s_am_ruido[:300],
    color='purple',
    alpha=0.8,
    label='AM + Ruido AWGN',
)
axs[1, 1].set_title('4. Señal AM con Ruido Blanco Gaussiano (Tiempo)')
axs[1, 1].set_xlabel('Tiempo (ms)')
axs[1, 1].set_ylabel('Amplitud (V)')
axs[1, 1].legend(loc='upper right')

# Gráfico 5: Distorsión por Sobremodulación (m = 1.5)
axs[2, 0].plot(
    t[:300] * 1000,
    s_am_distorsion[:300],
    color='crimson',
    label='Sobremodulación (m = 1.5)',
)
axs[2, 0].set_title('5. Distorsión por Sobremodulación (Tiempo)')
axs[2, 0].set_xlabel('Tiempo (ms)')
axs[2, 0].set_ylabel('Amplitud (V)')
axs[2, 0].legend(loc='upper right')

# Gráfico 6: Atenuación vs Señal Original
axs[2, 1].plot(
    t[:300] * 1000, s_am[:300], color='green', alpha=0.4, label='AM Original'
)
axs[2, 1].plot(
    t[:300] * 1000,
    s_am_atenuada[:300],
    color='navy',
    label='AM Atenuada (30%)',
)
axs[2, 1].set_title('6. Efecto de la Atenuación en el Canal (Tiempo)')
axs[2, 1].set_xlabel('Tiempo (ms)')
axs[2, 1].set_ylabel('Amplitud (V)')
axs[2, 1].legend(loc='upper right')

plt.show()

