# ensemble-learning
ensemble-learning


To run the app in streamlit cloud I needed to change the requirements.txt from

streamlit==1.31.0
pandas==2.2.0
numpy==1.26.3
scikit-learn==1.4.0
matplotlib==3.8.2
seaborn==0.13.2
xgboost==2.0.3



to 


streamlit>=1.31,<2
numpy>=2.1
pandas>=2.2.3
scikit-learn>=1.6
matplotlib>=3.8
seaborn>=0.13
xgboost>=2.0


Streamlit Community Cloud log shows it’s building with Python 3.13:

Using Python 3.13.12 ...

and hence I needed to change my dependencies

