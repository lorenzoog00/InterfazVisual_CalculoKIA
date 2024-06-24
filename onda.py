#Cálculo de onda kga
def calcular_kg(a, R, T, D_v, K5, V_w, mu_v, rho_v, d_p):
    # Ecuación de Onda para k_G
    term1 = (V_w / (a * mu_v)) ** 0.7
    term2 = (mu_v / (rho_v * D_v)) ** (1/3)
    term3 = (a * d_p) ** -2.0

    kg_a_RT_Dv = K5 * term1 * term2 * term3
    k_G = kg_a_RT_Dv * D_v*a / (R * T)

    return k_G