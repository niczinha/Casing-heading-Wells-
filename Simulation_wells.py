import casadi as ca
g = 9.81  # Aceleração da gravidade [m/s²]
R = 8.314 # Constante dos gases [J/mol.K]


M = 0.028 # Massa molar do gás [kg/mol]
ro_o = 800 # Densidade do óleo no reservatório [kg/m³] 
Ps = 2e6  # Pressão do separador [Pa]
vo = 1 / ro_o # Volume específico do óleo [m³/kg]

# Parâmetros do Riser
Dr = 0.121 # Diâmetro do riser [m]
Hr_riser_comum = 500 # Altura vertical do riser
Tr_riser_comum = 301 # Temperatura do riser [K] 
Crh = 10e-3 # Coeficiente da válvula da cabeça do riser [m²]
Ar_riser_comum = (ca.pi * (Dr ** 2)) / 4 # Área da seção transversal do riser [m²]
Ta = 28 + 273.15
# --- Parâmetros do Poço 1 ---
# Geometria e Temperaturas
Lw1 = 1500 # Comprimento do tubo [m]
Dw1 = 0.121 # Diâmetro do poço [m]
Hw1 = 1000 # Altura de coluna hidrostática no poço [m]
Hbh1 = 500
Lbh1 = 500 # Comprimento do poço abaixo do ponto de injeção [m]
Dbh1 = 0.121 # Diâmetro da seção abaixo do ponto injeção [m]
Tw1 = 305 # Temperatura no tubo [K]

La1 = 1500 # Comprimento da região anular [m]
Dr= 0.121
Da1 = 0.189 # Diâmetro do anular [m]
Ha1 = 1000 # Altura da região anular [m]
Ta1 = 301 # Temperatura da região anular [K]

# Coeficientes e Propriedades
PI1 = 0.7e-5 # Índice de Produtividade [kg/(s·Pa)]
Cpc1 = 2e-3 # Coeficiente da choke de produção [m²]
Civ1 = 0.1e-3 # Coeficiente da válvula de injeção [m²]
Pr1 = 1.50e7 # Pressão no reservatório [Pa]
Lr_poco1 = 500 # Distância do reservatório até o ponto de injeção 
GOR1 = 0.1# Razão Gás-Óleo [kg/kg] 39

# Áreas e Volumes Calculados para Poço 1
Aw1 = (ca.pi * (Dw1 ** 2)) / 4 # Área da seção transversal do poço [m²]
Aa1 = (ca.pi * (Da1 ** 2)) / 4 - (ca.pi * (Dw1 ** 2)) / 4 # Área da seção transversal do anel [m²]
Abh1 = (ca.pi * (Dbh1 ** 2)) / 4 # Área da seção transversal abaixo do ponto de injeção [m²]
Ar1 = (ca.pi * (Dr ** 2)) / 4 
Va1 = La1 * Aa1 # Volume da região anular [m³]

# --- Parâmetros do Poço 2 
# Geometria e Temperaturas
Lw2 = 1500
Lr2=500
Dw2 = 0.121
Hw2 = 1000
Lbh2 = 500
Hbh2 = 500
Dbh2 = 0.121
Tw2 = 305
Lr1=500
La2 = 1500
Da2 = 0.189
Dr= 0.121
Ha2 = 1000
Ta2 = 301
Cr = 2.6e-4
# Coeficientes e Propriedades
PI2 = 0.7e-5 # Pode ser diferente, exemplo: 0.23e-5
Cpc2 = 2e-3
Civ2 = 0.1e-3
Pr2 = 1.55e7 # Diferente do poço 1
Lr_poco2 = 500
GOR2 = 0.07

# Áreas e Volumes Calculados para Poço 2
Aw2 = (ca.pi * (Dw2 ** 2)) / 4
Aa2 = (ca.pi * (Da2 ** 2)) / 4 - (ca.pi * (Dw2 ** 2)) / 4
Abh2 = (ca.pi * (Dbh2 ** 2)) / 4
Ar2 = (ca.pi * (Dr ** 2)) / 4 
Va2 = La2 * Aa2


Lr_comum = 500 # Length do riser comum
Ar_comum = 0.0114 # Area do riser comum
Crh = 10E-3
Tr_comum = 30 + 273.15 # Temperatura média no riser em K

u1 = 1
u2 = 1
dp_fric_t1 = 0
dp_fric_bh1 = 0
dp_fric_t2 = 0
dp_fric_bh2 = 0
dp_fric_r_comum = 0


def fun(t, x, par):

    # Desempacota o vetor de 6 parâmetros
    wgl1, wgl2, u1, u2, GOR1, GOR2 = par[0], par[1], par[2], par[3], par[4], par[5]

    # Desempacota o vetor de 8 estados
    m_ga1, m_ga2, m_gt1, m_gt2, m_ot1, m_ot2, m_gr, m_or = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]

    # --- Poço 1 ---
    Pai1 = ((R * Ta1 / (Va1 * M)) + ((g * La1) / Va1)) * m_ga1
    ro_w1 = (m_gt1 + m_ot1) / (Lw1 * Aw1)
    Pwh1 = (R * Tw1 / M) * (m_gt1 / (Lw1 * Aw1 - (vo * m_ot1) + 1e-9))
    Pwi1 = Pwh1 + (g / Aw1) * (m_ot1 + m_gt1) 
    Pbh1 = Pwi1 + (ro_o * g * Hbh1) 
    ro_a1 = (M * Pai1) / (R * Ta)
    wiv1 = Civ1 * ca.sqrt(ca.fmax(1e-9, ro_a1 * (Pai1 - Pwi1)))
    y2_1 = ca.fmax(1e-9, (Pr1 - Pbh1))
    wro1 = Cr * ca.sqrt(ro_o * y2_1)
    wrg1 = GOR1 * wro1

    # --- Poço 2 ---
    Pai2 = ((R * Ta2 / (Va2 * M)) + ((g * La2) / Va2)) * m_ga2
    ro_w2 = (m_gt2 + m_ot2) / (Lw2 * Aw2)
    Pwh2 = (R * Tw2 / M) * (m_gt2 / (Lw2 * Aw2 - (vo * m_ot2) + 1e-9))
    ro_a2 = (M * Pai2) / (R * Ta)
    Pwi2 = Pwh2 + (g / Aw2) * (m_ot2 + m_gt2) + dp_fric_t2
    Pbh2 = Pwi2 + (ro_o * g * Hbh2) + dp_fric_bh2
    wiv2 = Civ2 * ca.sqrt(ca.fmax(1e-9, ro_a2 * (Pai2 - Pwi2)))
    y2_2 = ca.fmax(1e-9, (Pr2 - Pbh2))
    wro2 = Cr * ca.sqrt(ro_o * y2_2)
    wrg2 = GOR2 * wro2

    # --- Manifold e Riser Comum ---
    ro_r = (m_gr + m_or) / (Lr_comum * Ar_riser_comum)
    Prh = (Tr_comum * R / M) * (m_gr / (Lr_comum * Ar_riser_comum - vo * m_or + 1e-9))
    Pm = Prh + (ro_r * g * Hr_riser_comum) + dp_fric_r_comum

    # --- Vazões de Produção (Chokes) ---
    y3_1 = ca.fmax(1e-9, (Pwh1 - Pm))
    wpc1 = Cpc1 * ca.sqrt(ro_w1 * y3_1) * 50**(u1 - 1)
    wpg1_prod = (m_gt1 / (m_gt1 + m_ot1 + 1e-9)) * wpc1
    wpo1_prod = (m_ot1 / (m_gt1 + m_ot1 + 1e-9)) * wpc1

    y3_2 = ca.fmax(1e-9, (Pwh2 - Pm))
    wpc2 = Cpc2 * ca.sqrt(ro_w2 * y3_2) * 50**(u2 - 1)
    wpg2_prod = (m_gt2 / (m_gt2 + m_ot2 + 1e-9)) * wpc2
    wpo2_prod = (m_ot2 / (m_gt2 + m_ot2 + 1e-9)) * wpc2

    # --- Vazões de Saída do Riser ---
    y4_rh = ca.fmax(1e-9, (Prh - Ps))
    wrh = Crh * ca.sqrt(ro_r * y4_rh)
    wtg = (m_gr / ca.fmax(1e-9, m_gr + m_or)) * wrh
    wto = (m_or / ca.fmax(1e-9, m_gr + m_or)) * wrh

    # --- Derivadas dos Estados (Balanço de Massa) ---
    dx0 = wgl1 - wiv1          # Massa de gás no anular 1
    dx1 = wgl2 - wiv2          # Massa de gás no anular 2
    dx2 = wiv1 + wrg1 - wpg1_prod  # Massa de gás no tubing 1
    dx3 = wiv2 + wrg2 - wpg2_prod  # Massa de gás no tubing 2
    dx4 = wro1 - wpo1_prod         # Massa de óleo no tubing 1
    dx5 = wro2 - wpo2_prod         # Massa de óleo no tubing 2
    dx6 = wpg1_prod + wpg2_prod - wtg # Massa de gás no riser
    dx7 = wpo1_prod + wpo2_prod - wto # Massa de óleo no riser

    return ca.vertcat(dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7)

def modelo(x, par):
    """
    Calcula as variáveis de saída do sistema.
    """
    # Desempacota o vetor de 6 parâmetros
    wgl1, wgl2, u1, u2, GOR1, GOR2 = par[0], par[1], par[2], par[3], par[4], par[5]

    # Desempacota o vetor de 8 estados
    m_ga1, m_ga2, m_gt1, m_gt2, m_ot1, m_ot2, m_gr, m_or = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]

    # --- Poço 1 ---
    Pai1 = ((R * Ta1 / (Va1 * M)) + ((g * La1) / Va1)) * m_ga1
    ro_w1 = (m_gt1 + m_ot1) / (Lw1 * Aw1)
    Pwh1 = (R * Tw1 / M) * (m_gt1 / (Lw1 * Aw1 - (vo * m_ot1) + 1e-9))
    Pwi1 = Pwh1 + (g / Aw1) * (m_ot1 + m_gt1) + dp_fric_t1
    Pbh1 = Pwi1 + (ro_o * g * Hbh1) + dp_fric_bh1
    
    # --- Poço 2 ---
    Pai2 = ((R * Ta2 / (Va2 * M)) + ((g * La2) / Va2)) * m_ga2
    ro_w2 = (m_gt2 + m_ot2) / (Lw2 * Aw2)
    Pwh2 = (R * Tw2 / M) * (m_gt2 / (Lw2 * Aw2 - (vo * m_ot2) + 1e-9))
    Pwi2 = Pwh2 + (g / Aw2) * (m_ot2 + m_gt2) + dp_fric_t2
    Pbh2 = Pwi2 + (ro_o * g * Hbh2) + dp_fric_bh2

    # --- Manifold e Riser Comum ---
    ro_r = (m_gr + m_or) / (Lr_comum * Ar_riser_comum)
    Prh = (Tr_comum * R / M) * (m_gr / (Lr_comum * Ar_riser_comum - vo * m_or + 1e-9))
    Pm = Prh + (ro_r * g * Hr_riser_comum) + dp_fric_r_comum

    # --- Vazões de Produção (Chokes) ---
    y3_1 = ca.fmax(1e-9, (Pwh1 - Pm))
    wpc1 = Cpc1 * ca.sqrt(ro_w1 * y3_1) * 50**(u1 - 1)
    wpo1_prod = (m_ot1 / (m_gt1 + m_ot1 + 1e-9)) * wpc1

    y3_2 = ca.fmax(1e-9, (Pwh2 - Pm))
    wpc2 = Cpc2 * ca.sqrt(ro_w2 * y3_2) * 50**(u2 - 1)
    wpo2_prod = (m_ot2 / (m_gt2 + m_ot2 + 1e-9)) * wpc2

    # --- Vazões de Saída do Riser ---
    y4_rh = ca.fmax(1e-9, (Prh - Ps))
    wrh = Crh * ca.sqrt(ro_r * y4_rh)
    wto = (m_or / ca.fmax(1e-9, m_gr + m_or)) * wrh
    
    return {
        'wpo1': wpo1_prod, 'wpo2': wpo2_prod, 'wto_riser': wto, 
        'wpc1': wpc1, 'wpc2': wpc2, 'Pbh1': Pbh1, 'Pbh2': Pbh2, 
        'Pm': Pm, 'Prh': Prh
    }
