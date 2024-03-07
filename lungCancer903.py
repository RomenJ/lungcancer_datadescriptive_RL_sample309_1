import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
#dataset: https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer

def load_data(file_path):
    """Cargar datos desde un archivo CSV"""
    return pd.read_csv(file_path)


def visualize_swarmplot(df, y, x, hue):
    """Visualizar swarmplot"""
    plt.title('swarmplot')
    sns.swarmplot(data=df, y=y, x=x, hue=hue)
    plt.savefig(f'swarmplot {hue}.png')
    plt.show()


def print_data_summary(df):
    """Imprimir resumen de datos"""
    print(df.head(5))
    print(df.columns)
    print(df.info)
    print("Shape: ", df.shape)
    print("Tipos de valores 01")
    print(df.dtypes)


def visualize_heatmap(correlation_matrix):
    """Visualizar mapa de calor de correlaciones"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlaciones entre las características')
    plt.savefig('Heatmap.png')
    plt.show()


def visualize_distribution(df, columns):
    """Visualizar distribución de frecuencias de variables numéricas"""
    for column in columns:
        sns.histplot(df[column], bins=20, kde=True)
        plt.title(f'Distribución de {column}')
        plt.savefig(f'Distribución de {column}.png')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        plt.show()


def visualize_target_distribution(df, target_column):
    """Visualizar distribución de la variable objetivo"""
    sns.countplot(data=df, x=target_column)
    plt.title(f'Distribución de {target_column}')
    plt.xlabel(target_column)
    plt.ylabel('Frecuencia')
    plt.savefig(f'Distribución de {target_column}.png')
    plt.show()


def preprocess_data(df):
    """Preprocesar los datos"""
    df['LUNG_CANCER_NUM'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    df['GENDER_NUM'] = df['GENDER'].map({'M': 2, 'F': 1})
    return df


def select_numeric_columns(df):
    """Seleccionar columnas numéricas"""
    return df.select_dtypes(include='number').columns


def train_logistic_regression(X_train, y_train):
    """Entrenar el modelo de regresión logística"""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluar el modelo"""
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, (y_proba > 0.5).astype(int))
    print("Precisión del modelo:", accuracy)

    conf_matrix = confusion_matrix(y_test, (y_proba > 0.5).astype(int))
    print("Matriz de confusión:")
    print(conf_matrix)

    roc_auc = roc_auc_score(y_test, y_proba)
    print("Puntuación ROC AUC:", roc_auc)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    return fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc):
    """Graficar la curva ROC"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig('curva_ROC.png')
    plt.show()


def main():
    # Cargar datos
    file_path = "survey lung cancer.csv"
    df = load_data(file_path)
    df2 = load_data(file_path)
    print("DataFrameColumns",df.columns)
    # Resumen de datos
    print_data_summary(df)

    # Visualización de datos
    visualize_swarmplot(df, "LUNG_CANCER", "AGE", "SMOKING")

    visualize_swarmplot(df, "LUNG_CANCER", "AGE", "ALCOHOL CONSUMING")

    # Preprocesamiento de datos
    df = preprocess_data(df)
    print("Di DF",  df.columns)
    # Selección de columnas numéricas
    numeric_cols = select_numeric_columns(df)

    # Visualización de la distribución de variables numéricas
    visualize_distribution(df, numeric_cols)

    # Visualización de la distribución de la variable objetivo
    visualize_target_distribution(df, 'LUNG_CANCER')

    # Cálculo de correlaciones
    correlation_matrix = df[numeric_cols].corr()

    # Visualización de mapa de calor de correlaciones
    visualize_heatmap(correlation_matrix)

    # Selección de características y variable objetivo
    X = df[['ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'WHEEZING', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'YELLOW_FINGERS', "ALLERGY " ]]
    y = df['LUNG_CANCER_NUM']

    # División de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento del modelo
    model = train_logistic_regression(X_train, y_train)

    # Evaluación del modelo
    fpr, tpr, roc_auc = evaluate_model(model, X_test, y_test)

    # Visualización de la curva ROC
    plot_roc_curve(fpr, tpr, roc_auc)


if __name__ == "__main__":
    main()
