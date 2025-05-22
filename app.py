import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from egt_forecast import train_and_predict

st.set_page_config(
    page_title="Prédiction EGT Margin",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Prédiction EGT Margin")

# Sidebar pour le chargement du fichier
st.sidebar.header("Chargement des données")
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier Excel", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Sauvegarder temporairement le fichier
        with open("temp_data.xlsx", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Charger les données
        df = pd.read_excel("temp_data.xlsx")
        
        # Afficher les données brutes
        st.subheader("Données brutes")
        st.dataframe(df.head(), use_container_width=True)
        
        # Afficher le graphique des données brutes
        st.subheader("Visualisation des données brutes")
        fig_raw = go.Figure()
        
        # Ajouter la ligne des données brutes
        fig_raw.add_trace(
            go.Scatter(
                x=df['CSN'] if 'CSN' in df.columns else df.index,
                y=df['EGT Margin'],
                mode='lines+markers',
                name='EGT Margin',
                line=dict(color='blue'),
                marker=dict(size=4)
            )
        )
        
        # Ajouter les lignes de la zone critique
        fig_raw.add_trace(
            go.Scatter(
                x=df['CSN'] if 'CSN' in df.columns else df.index,
                y=[12] * len(df),
                mode='lines',
                name='Limite inférieure (12°C)',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig_raw.add_trace(
            go.Scatter(
                x=df['CSN'] if 'CSN' in df.columns else df.index,
                y=[18] * len(df),
                mode='lines',
                name='Limite supérieure (18°C)',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Mise à jour du layout pour les données brutes
        fig_raw.update_layout(
            title="Données brutes EGT Margin",
            xaxis_title="CSN",
            yaxis_title="EGT Margin (°C)",
            yaxis_range=[10, 50],
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_raw, use_container_width=True)
        
        # Bouton pour lancer la prédiction
        if st.button("Lancer la prédiction"):
            with st.spinner("Calcul des prédictions en cours..."):
                # Générer les prédictions
                forecast_df, metrics = train_and_predict("temp_data.xlsx")
                
                # Afficher les métriques
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R²", f"{metrics['r2']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}°C")
                with col3:
                    st.metric("MAE", f"{metrics['mae']:.4f}°C")
                
                # Afficher le graphique des prédictions
                st.subheader("Prévisions EGT Margin")
                fig = go.Figure()
                
                # Ajouter la ligne des prédictions
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df['CSN'],
                        y=forecast_df['Predicted EGT Margin'],
                        mode='lines',
                        name='Prédictions',
                        line=dict(color='blue')
                    )
                )
                
                # Ajouter une zone pour la zone critique (12-18°C)
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df['CSN'],
                        y=[12] * len(forecast_df),
                        mode='lines',
                        name='Limite inférieure (12°C)',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df['CSN'],
                        y=[18] * len(forecast_df),
                        mode='lines',
                        name='Limite supérieure (18°C)',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                # Mise à jour du layout
                fig.update_layout(
                    title="Prévision EGT Margin - Zone Rouge [12°C – 18°C]",
                    xaxis_title="CSN",
                    yaxis_title="EGT Margin (°C)",
                    yaxis_range=[10, 50],
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Afficher le tableau des prédictions
                st.subheader("Tableau des prédictions")
                st.dataframe(forecast_df, use_container_width=True)
                
                # Bouton de téléchargement des prédictions
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Télécharger les prédictions (CSV)",
                    data=csv,
                    file_name="predictions_egt_margin.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"Une erreur est survenue : {str(e)}")
    
    finally:
        # Nettoyage du fichier temporaire
        import os
        if os.path.exists("temp_data.xlsx"):
            os.remove("temp_data.xlsx")
else:
    st.info("Veuillez charger un fichier Excel pour commencer.") 