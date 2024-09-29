import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import streamlit as st

# Judul aplikasi
st.title("E-commerce Customer Data Analysis")

# Membaca file CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Menampilkan data frame
    st.write("### Dataset Preview", df.head())

    # Statistik deskriptif
    st.write("### Descriptive Statistics")
    st.write(df.describe())
    
    # Menghitung Interquartile Range (IQR)
    st.write("### Menghitung Interquartile Range (IQR)")
    q1 = df.select_dtypes(exclude=['object']).quantile(0.25)
    q3 = df.select_dtypes(exclude=['object']).quantile(0.75)
    iqr = q3 - q1
    st.write(iqr)
    
    # Menghitung Batas Bawah
    st.write("### Menghitung Batas Bawah")
    batas_bawah = q1 - (1.5 * iqr)
    st.write(batas_bawah)
    
    # Menghitung Batas Atas
    st.write("### Menghitung Batas Atas")
    batas_atas = q3 + (1.5 * iqr)
    st.write(batas_atas)

    # Filter outlier pada dataset
    outlier_filter = (df.select_dtypes(exclude=['object']) < q1 - 1.5 * iqr) | (df.select_dtypes(exclude=['object']) > q3 + 1.5 * iqr)

    # Menghitung Jumlah Outlier Pada Data
    for col in outlier_filter.columns:
        if df[col].dtype != object:
            st.write('Nama Kolom:', col)
            st.write(outlier_filter[col].value_counts())
            st.write('-------------------')

    # Mencari Outliers - Boxplot
    st.write("### Boxplot untuk Outliers")
    for column in df.select_dtypes(exclude=['object']).columns:
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.boxplot(data=df, x=column, ax=ax)
        ax.set_title(f'Boxplot of {column}')
        st.pyplot(fig)
        plt.close(fig)  # Close figure to avoid conflicts
    
    # Melihat Persebaran Data - Histogram
    st.write("### Histogram Data")
    for column in df.select_dtypes(exclude=['object']).columns:
        fig, ax = plt.subplots()
        ax.hist(df[column], bins=20)
        ax.set_title(f'Histogram of {column}')
        st.pyplot(fig)
        plt.close(fig)

    # Scatter plot: Time on App vs Yearly Amount Spent
    st.write("### Scatter plot of Time on App vs Yearly Amount Spent")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Time on App', y='Yearly Amount Spent', ax=ax)
    ax.set_title('Time on App vs Yearly Amount Spent')
    st.pyplot(fig)

    # Scatter plot: Time on Website vs Yearly Amount Spent
    st.write("### Scatter plot of Time on Website vs Yearly Amount Spent")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Time on Website', y='Yearly Amount Spent', ax=ax)
    ax.set_title('Time on Website vs Yearly Amount Spent')
    st.pyplot(fig)

    # Pairplot
    st.write("### Pairplot")
    fig = sns.pairplot(df.select_dtypes(exclude=['object']))
    st.pyplot(fig)
    plt.close()  # Close the figure

    # lmplot: Length of Membership vs Yearly Amount Spent
    st.write("### Linear Model Plot")
    fig = sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df)
    st.pyplot(fig)
    plt.close()  # Close the figure

    # Data preparation
    st.write("### Data Preparation")
    st.write("#### Melihat data bernilai NULL")
    st.code("df.isna().sum())")
    st.write(df.isna().sum())
    
    # Modelling information
    st.write("### Modelling")
    st.write("#### Membuat Variabel dependen dan independen")
    st.code("X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']] y = df['Yearly Amount Spent']")
    X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y = df['Yearly Amount Spent']
    
    st.code("X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
    st.code("lr = LinearRegression()")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    st.write("### Melihat hasil koefisien regresi")
    st.write(lr.coef_)
    st.write("### R SQUARED")
    st.write(lr.score(X, y))
    st.write("### Melihat koefisien regresi dari masing masing tabel")
    cdf = pd.DataFrame(lr.coef_,X.columns,columns=['Coef'])
    st.write(cdf)
    predictions = lr.predict(X_test)
    st.write("Prediction for test set: {}".format(predictions))
    st.write("### Membandingkan nilai sebenarnya dengan nilai prediksi menggunakan Linear Regression")
    lr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': predictions})
    st.write(lr_diff.head())
    st.write("### Scatter plot of Actual values vs Predicted values")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=predictions, ax=ax)
    ax.set_ylabel('Predicted Values')
    ax.set_xlabel('Actual Values')
    ax.set_title('Yearly Amount Spent vs Model Predictions')
    st.pyplot(fig)
    plt.close(fig)
    
    st.write("# Model Evaluation")
    st.write('Mean Absolute Error:',mean_absolute_error(y_test, predictions))
    st.write('Mean Squared Error:',mean_squared_error(y_test, predictions))
    st.write('Root Mean Squared Error:',math.sqrt(mean_squared_error(y_test, predictions)))
