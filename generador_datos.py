import pandas as pd
import numpy as np

np.random.seed(42)

# 1. Fechas
fechas = pd.date_range(start='2024-01-01', end='2026-02-28', freq='D')
df = pd.DataFrame({'Fecha': fechas})
df['Dia'] = df['Fecha'].dt.day
df['Mes'] = df['Fecha'].dt.month
df['Dia_Semana'] = df['Fecha'].dt.dayofweek # 0=Lunes, 6=Domingo

# Inicializamos columnas
df['Ingresos'] = 0.0
df['Gastos'] = 0.0

# 2. LÓGICA DE INGRESOS (El modelo híbrido)
meses_curso = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6]

for idx, row in df.iterrows():
    mes = row['Mes']
    dia = row['Dia']
    dia_semana = row['Dia_Semana']
    
    if mes in meses_curso:
        if dia <= 5: # Cobro de mensualidades
            df.at[idx, 'Ingresos'] = np.random.normal(450000, 50000)
        else:
            # Resto del mes: ingresos residuales (vending, lavandería, eventos)
            df.at[idx, 'Ingresos'] = np.random.normal(3000, 500)
    else:
        if dia_semana >= 4: # Fines de semana de verano
            df.at[idx, 'Ingresos'] = np.random.normal(95000, 10000)
        else:
            df.at[idx, 'Ingresos'] = np.random.normal(65000, 8000)

# 3. LÓGICA DE GASTOS
df['Gastos'] = np.where(df['Mes'].isin([7, 8]), 
                        np.random.normal(25000, 2000, len(df)),  # Gastos verano
                        np.random.normal(15000, 1500, len(df)))  # Gastos  resto

# 4. HITOS FINANCIERS Y FISCALES
# NÓMINAS (Día 28)
df.loc[df['Dia'] == 28, 'Gastos'] += 350000

# DEUDA / LEASING (Día 5)
df.loc[df['Dia'] == 5, 'Gastos'] += 1200000

# IMPUESTOS / IVA (Día 20 del mes siguiente al trimestre)
meses_iva = [1, 4, 7, 10]
df.loc[(df['Mes'].isin(meses_iva)) & (df['Dia'] == 20), 'Gastos'] += 400000

# EL DIVIDENDO
# Salida masiva de caja el 30 de junio para retribuir a los accionistas
df.loc[(df['Mes'] == 6) & (df['Dia'] == 30), 'Gastos'] += 1000000

# 5. CÁLCULO DE CAJA Y SALDO
df['CashFlow'] = df['Ingresos'] - df['Gastos']
df['Saldo_Bancario'] = df['CashFlow'].cumsum() + 3200000 # Empezamos con 3.2M€ de colchón

# Exportación de la información a CSV
df_final = df[['Fecha', 'Ingresos', 'Gastos', 'CashFlow', 'Saldo_Bancario']].round(2)
df_final.to_csv('tesoreria_r.csv', index=False)
print("✅ Archivo 'tesoreria_r.csv' generado con márgenes reales y dividendos.")
