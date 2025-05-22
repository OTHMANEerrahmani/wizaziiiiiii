import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xlsxwriter

def train_and_predict(file_path: str):
    """
    Entraîne le modèle et génère les prédictions.
    
    Args:
        file_path (str): Chemin vers le fichier Excel d'entrée
        
    Returns:
        tuple: (DataFrame des prédictions, dictionnaire des métriques)
    """
    # Charger les données
    df = pd.read_excel(file_path)
    
    # Générer les 30 lags
    for lag in range(1, 31):
        df[f'lag_{lag}'] = df['EGT Margin'].shift(lag)
    
    df_lagged = df.dropna().reset_index(drop=True)
    
    # Définir X et y
    X = df_lagged[[f'lag_{i}' for i in range(1, 31)]]
    y = df_lagged['EGT Margin']
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Modèle
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Générer prédictions pour 200 cycles
    last_30 = df['EGT Margin'].iloc[-30:].tolist()
    future_preds = []
    for _ in range(200):
        input_array = np.array(last_30[-30:]).reshape(1, -1)
        pred = model.predict(input_array)[0]
        future_preds.append(pred)
        last_30.append(pred)
    
    # Créer le DataFrame des prédictions
    csn_start = df['CSN'].iloc[-1] + 1 if 'CSN' in df.columns else 22576
    forecast_df = pd.DataFrame({
        'CSN': np.arange(csn_start, csn_start + 200),
        'Predicted EGT Margin': future_preds
    })
    
    # Créer le dictionnaire des métriques
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }
    
    return forecast_df, metrics

if __name__ == "__main__":
    # Code pour l'exécution directe du script
    forecast_df, metrics = train_and_predict("802970 ready to use data.xlsx")
    
    # Export Excel avec zone colorée
    output_file = "EGT_Margin_Forecast_Zone_Rouge.xlsx"
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        forecast_df.to_excel(writer, index=False, sheet_name='Prévision EGT')
        metrics_df = pd.DataFrame({
            'Metric': ['Test R²', 'Test RMSE (°C)', 'Test MAE (°C)'],
            'Value': [metrics['r2'], metrics['rmse'], metrics['mae']]
        })
        metrics_df.to_excel(writer, index=False, sheet_name='Performance')

        workbook = writer.book
        worksheet = writer.sheets['Prévision EGT']

        chart = workbook.add_chart({'type': 'line'})

        # Série principale
        chart.add_series({
            'name': 'EGT Prédit',
            'categories': ['Prévision EGT', 1, 0, 200, 0],
            'values':     ['Prévision EGT', 1, 1, 200, 1],
            'line': {'color': 'blue'}
        })

        # Zone critique (12°C < EGT < 18°C)
        chart.add_series({
            'name': 'Zone critique (12–18°C)',
            'categories': ['Prévision EGT', 1, 0, 200, 0],
            'values':     ['Prévision EGT', 1, 2, 200, 2],
            'line': {'color': 'red'}
        })

        chart.set_title({'name': 'Prévision EGT Margin - Zone Rouge [12°C – 18°C]'})
        chart.set_x_axis({'name': 'CSN'})
        chart.set_y_axis({'name': 'EGT Margin (°C)'})
        chart.set_legend({'position': 'bottom'})

        worksheet.insert_chart('E4', chart)

    print(f"✅ Fichier avec graphique et zone critique exporté : {output_file}") 