from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from onda import calcular_kG_principal

app = Flask(__name__)

# Load the scaler and model
scaler_X = joblib.load('minmax_scaler_X.pkl')
scaler_y = joblib.load('minmax_scaler_y.pkl')
model = load_model('best_model_2.4367305027108534e-05.h5')

# Cargar datos del archivo Excel
data = pd.read_excel('matriz_datos.xlsx', sheet_name='SOLO DATOS')

# Cargar datos de empaques
designed_data_for_various_packing = pd.read_excel('design_data_for_various_packings.xlsx')

def find_mea_range(mea_value, data):
    mea_concentrations = data['MEA wt%'].unique()
    lower_mea = max([mea for mea in mea_concentrations if mea <= mea_value], default=None)
    higher_mea = min([mea for mea in mea_concentrations if mea > mea_value], default=None)
    return lower_mea, higher_mea

def interpolate_y_relacion_molar(mea_value, lower_mea, higher_mea, data):
    if lower_mea is None or higher_mea is None:
        raise ValueError("No valid MEA range found.")
    
    lower_data = data[data['MEA wt%'] == lower_mea]["X Relación Molar 25°C"].tolist()
    higher_data = data[data['MEA wt%'] == higher_mea]["X Relación Molar 25°C"].tolist()
    ejeY = data[data['MEA wt%'] == lower_mea]["Y Relación Molar"].tolist()
    lower_data40 = data[data['MEA wt%'] == lower_mea]["X Relación Molar 40°C"].tolist()
    higher_data40 = data[data['MEA wt%'] == higher_mea]["X Relación Molar 40°C"].tolist()

    if not lower_data or not higher_data:
        raise ValueError("No data found for the given MEA ranges.")

    y_interpolado = []
    y_interpolado40 = []

    for i in range(len(higher_data)):
        y1 = lower_data[i]  # X a 25°C de lower_data
        y2 = higher_data[i]  # X a 25°C de higher_data

        # Fórmula de interpolación lineal
        y = y1 + ((y2 - y1) / (higher_mea - lower_mea)) * (mea_value - lower_mea)
        y_interpolado.append(y)

        y1_40 = lower_data40[i]  # X a 40 de lower_data
        y2_40 = higher_data40[i]  # X a 40 de higher_data

        # Fórmula de interpolación lineal
        y40 = y1_40 + ((y2_40 - y1_40) / (higher_mea - lower_mea)) * (mea_value - lower_mea)
        y_interpolado40.append(y40)

    return y_interpolado, y_interpolado40, ejeY

def interpolate_temperature(y_interpolado20, y_interpolado40, temperature, data):
    if temperature > 25:
        y_interpolado = []

        for i in range(len(y_interpolado20)):
            y1 = y_interpolado20[i]  # X a 25°C de lower_data
            y2 = y_interpolado40[i]  # X a 40°C de higher_data
            higher_temp = 40
            lower_temp = 25
            # Fórmula de interpolación lineal
            y = y1 + ((y2 - y1) / (higher_temp - lower_temp)) * (temperature - lower_temp)
            y_interpolado.append(y)
        return y_interpolado
    else:
        return y_interpolado20

def interpolate(concentration, temp):
    lower_mea, higher_mea = find_mea_range(concentration, data)
    inter_20, inter_40, ejeY = interpolate_y_relacion_molar(concentration, lower_mea, higher_mea, data)
    final_value = interpolate_temperature(inter_20, inter_40, temp, data)
    return ejeY, final_value

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/neural_network')
def neural_network():
    return render_template('neural_network.html')

@app.route('/cornell')
def cornell():
    return render_template('cornell.html')

@app.route('/shulman')
def shulman():
    return render_template('shulman.html')

@app.route('/onda')
def onda():
    materiales = designed_data_for_various_packing['Material'].unique().tolist()
    return render_template('onda.html', materiales=materiales)

@app.route('/get_sizes', methods=['POST'])
def get_sizes():
    material = request.json['material']
    sizes = designed_data_for_various_packing[designed_data_for_various_packing['Material'] == material]['Size (in.)'].unique().tolist()
    return jsonify(sizes)

@app.route('/calcular', methods=['POST'])
def calcular():
    material = request.form['material']
    size_in = float(request.form['size_in'])
    T = float(request.form['T'])
    flujo_masico = float(request.form['flujo_masico'])
    y1 = float(request.form['y1'])
    K5 = float(request.form['K5'])

    kga = calcular_kG_principal(material, size_in, T, flujo_masico, y1, K5)
    return render_template('resultado.html', kga=kga)


@app.route('/ann', methods=['GET', 'POST'])
def ann():
    if request.method == 'POST':
        # Extract data from form
        y1 = float(request.form['y1'])
        G_ = float(request.form['G_'])
        wt_mea = float(request.form['wt_mea'])
        z = float(request.form['z'])
        T_gas = float(request.form['T_gas'])
        L_ = float(request.form['L_'])
        
        # Create a DataFrame for the inputs
        input_data = pd.DataFrame([[y1, G_, wt_mea, z, T_gas, L_]], columns=['y1', 'G_', 'wt_mea', 'z', 'T_gas', 'L_'])
        
        # Normalize the input data
        input_data_normalized = scaler_X.transform(input_data)
        
        # Predict using the loaded model
        prediction_normalized = model.predict(input_data_normalized)
        
        # Inverse transform the prediction to get the original scale
        prediction = scaler_y.inverse_transform(prediction_normalized)
        kga = float(prediction[0][0])  # Convert to native float
        
        # Return the result
        return render_template('ann.html', kga=kga)
    
    return render_template('ann.html')

@app.route('/mea_interpolation', methods=['GET', 'POST'])
def mea_interpolation():
    if request.method == 'POST':
        try:
            concentration = float(request.form['concentration'])
            temp = float(request.form['temperature'])
            
            x, y = interpolate(concentration, temp)
            
            # Esas son las variables x e y que quiero importar
            plt.figure()
            plt.plot(y, x, '-o', label='Curva de equilibrio')
            plt.xlabel('X Relación Molar')
            plt.ylabel('Y Relación Molar')
            plt.title(f'Equilibrio de concentración de MEA a {concentration}% y {temp}°C')
            plt.legend()
            
            # Convertir la gráfica a una imagen base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

            return render_template('mea_interpolation.html', plot_url=plot_url, concentration=concentration, temperature=temp, x=x, y=y)
        except Exception as e:
            return render_template('mea_interpolation.html', error=str(e))

    return render_template('mea_interpolation.html')

@app.route('/download/csv')
def download_csv():
    concentration = request.args.get('concentration')
    temp = request.args.get('temperature')
    x = request.args.getlist('x', type=float)
    y = request.args.getlist('y', type=float)
    
    file_path = f'tem_{temp}_conc_{concentration}.csv'
    df = pd.DataFrame({'Y Relación Molar': y, 'X Relación Molar': x})
    df.to_csv(file_path, index=False)
    return send_file(file_path, as_attachment=True, mimetype='text/csv')

@app.route('/download/txt')
def download_txt():
    concentration = request.args.get('concentration')
    temp = request.args.get('temperature')
    x = request.args.getlist('x', type=float)
    y = request.args.getlist('y', type=float)
    
    file_path = f'tem_{temp}_conc_{concentration}.txt'
    with open(file_path, 'w') as f:
        for xi, yi in zip(x, y):
            f.write(f"{yi}\t{xi}\n")
    return send_file(file_path, as_attachment=True, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)