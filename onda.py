import pandas as pd
import numpy as np

# Cargar los datos desde el archivo Excel
designed_data_for_various_packing = pd.read_excel('design_data_for_various_packings.xlsx')

# Constantes
R = 8.314  # J/(mol*K), constante de los gases ideales
PMaire = 28.97  # g/mol, peso molecular del aire
PMco2 = 44.01  # g/mol, peso molecular del CO2
o_ab = 3.65  # A, parámetro de difusión
ΩD = 1.0  # factor de difusión, depende de la temperatura
P = 101325  # Pa, presión estándar

def pulgadas_a_milimetros(pulgadas):
    """Convierte pulgadas a milímetros"""
    return pulgadas * 25.4

def seleccionar_valores(material, size_in):
    """Selecciona los valores de a y d_p basados en el material y el tamaño en pulgadas"""
    size_mm = pulgadas_a_milimetros(size_in)
    filtro = (designed_data_for_various_packing['Material'] == material) & (designed_data_for_various_packing['Size (in.)'] == size_in)
    fila = designed_data_for_various_packing[filtro].iloc[0]
    a = fila['Surface area (m^2/m^3)']
    d_p = fila['Size (mm)']
    return a, d_p

def calcular_kg(a, R, T, D_v, K5, V_w, mu_v, rho_v, d_p):
    """Calcula el coeficiente de transferencia de masa kG"""
    term1 = (V_w / (a * mu_v)) ** 0.7
    term2 = (mu_v / (rho_v * D_v)) ** (1/3)
    term3 = (a * d_p) ** -2.0

    kg_a_RT_Dv = K5 * term1 * term2 * term3
    k_G = kg_a_RT_Dv * D_v * a / (R * T)

    return k_G

def calcular_kG_principal(material, size_in, T, flujo_masico, y1, K5):
    """Función principal que toma las variables de entrada y calcula kG"""
    a, d_p = seleccionar_valores(material, size_in)

    # Difusión
    D_v = (((0.001858 * T ** (3 / 2)) * ((PMaire + PMco2) / (PMaire * PMco2)) ** 0.5) / (P * (o_ab ** 2) * ΩD)) / 1e4

    # Viscosidades
    mu_aire = 5E-08 * T + 2E-05
    mu_co2 = 5E-08 * T + 1E-05

    term1 = (1 / np.sqrt(8)) * (1 + PMco2 / PMaire) ** (-1 / 2)
    term2 = (1 + (np.sqrt(mu_co2 / mu_aire)) * (PMco2 / PMaire) ** (1 / 4)) ** 2
    Phi_ac = term1 * term2

    term1 = (1 / np.sqrt(8)) * (1 + PMaire / PMco2) ** (-1 / 2)
    term2 = (1 + (np.sqrt(mu_aire / mu_co2)) * (PMaire / PMco2) ** (1 / 4)) ** 2
    Phi_ca = term1 * term2

    # Viscosidad de Mezcla
    term3 = y1 * mu_co2 / (y1 + (1 - y1) * Phi_ca)
    term4 = (1 - y1) * mu_aire / ((1 - y1) + y1 * Phi_ac)
    mu_v = term3 + term4  # [Pa*s o Ns/m2]

    # Densidad
    PMmezcla = y1 * PMco2 + (1 - y1) * PMaire
    rho_v = P * PMmezcla / (R * T)

    # Flux volumétrico
    V_w = flujo_masico / rho_v  # Convertir flujo másico a volumétrico

    # Calcular kG
    kG_v = float(calcular_kg(a, R, T, D_v, K5, V_w, mu_v, rho_v, d_p))

    return kG_v