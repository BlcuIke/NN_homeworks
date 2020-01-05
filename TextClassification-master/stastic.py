from sklearn.metrics import classification_report
import pandas as pd

with open('predictions.csv', 'r', encoding='utf-8') as inp:
    c = pd.read_csv(inp)

print(classification_report(c['True class'], c['Prediction']))
