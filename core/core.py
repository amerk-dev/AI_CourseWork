import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('../data/german_credit_data.csv')

# Handle missing values by filling with a placeholder or a statistical value
df = df.copy()  # Ensure df is a copy and avoid chained assignment issues
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

# Encode categorical variables
label_encoders = {}
for column in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

df = df.replace([np.inf, -np.inf], np.nan)

# Age distribution
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Define features and target
X = df.drop(columns=['Credit amount'])  # Assuming 'Credit amount' as target
y = df['Credit amount'] > df['Credit amount'].median()  # Binary classification (High/Low Credit Amount)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
# Credit amount distribution
sns.histplot(df['Credit amount'], kde=True)
plt.title('Credit Amount Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

print(df.head())

# Add predictions to the DataFrame
df_test = df.iloc[y_test.index].copy()  # Create a copy of the test dataset rows from the original DataFrame
df_test['Prediction'] = y_pred
df_test['Actual'] = y_test.values

# True Negatives: Actual = 0 (low risk), Prediction = 0 (low risk)
true_negatives = df_test[(df_test['Actual'] == 0) & (df_test['Prediction'] == 0)]

# True Positives: Actual = 1 (high risk), Prediction = 1 (high risk)
true_positives = df_test[(df_test['Actual'] == 1) & (df_test['Prediction'] == 1)]

# Display True Negatives
print("True Negatives (Low risk correctly predicted as low risk):")
print(true_negatives)

# Display True Positives
print("\nTrue Positives (High risk correctly predicted as high risk):")
print(true_positives)


def predict_credit_risk():
    try:
        user_data = {}
        print("\nВведите данные клиента для оценки кредитного риска:")
        user_data['Age'] = int(input("Возраст: "))
        user_data['Sex'] = input("Пол (male/female): ").lower()
        user_data['Job'] = int(
            input("Профессия (0-безработный, 1-неквалифицированный, 2-квалифицированный, 3-высококвалифицированный): "))
        user_data['Housing'] = input("Жильё (own/rent/free): ").lower()
        user_data['Saving accounts'] = input("Сберегательный счёт (little/moderate/quite rich/rich/unknown): ").lower()
        user_data['Checking account'] = input("Текущий счёт (little/moderate/quite rich/rich/unknown): ").lower()
        user_data['Duration'] = int(input("Срок кредита (в месяцах): "))
        user_data['Purpose'] = input("Цель кредита: ").strip().lower()  # Изменение здесь

        # Обработка категориальных переменных
        for column in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
            le = label_encoders[column]
            value = user_data[column]

            # Проверка допустимых значений
            if value not in le.classes_:
                allowed_values = list(le.classes_)
                raise ValueError(f"Недопустимое значение для '{column}'. Допустимые значения: {allowed_values}")

            user_data[column] = le.transform([value])[0]

        # Создание DataFrame
        input_df = pd.DataFrame([user_data], columns=X.columns)

        # Масштабирование признаков
        scaled_input = scaler.transform(input_df)

        # Предсказание
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0]

        # Вывод результата
        print("\nРезультат оценки кредитного риска:")
        print(f"Прогноз: {'Высокий риск' if prediction[0] else 'Низкий риск'}")
        print(f"Вероятность высокого риска: {probability[1] * 100:.1f}%")

    except ValueError as e:
        print(f"\nОшибка ввода: {e}")


# Вызов функции предсказания
predict_credit_risk()