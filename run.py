# %% [markdown]
# # Utilizing Cashflow Underwriting for Fairer Credit Assessment

# %%
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel,DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import os
import torch
from datasets import Dataset

# %%
inflows = pd.read_parquet('data/q1-ucsd-inflows.pqt')
outflows = pd.read_parquet('data/q1-ucsd-outflows.pqt')

# %% [markdown]
# ## Data Exploration

# %%
in_transactions = len(inflows)
out_transactions = len(outflows)
print(f'Transactions in Inflows: {in_transactions}')
print(f'Transactions in Outflows: {out_transactions}')

# %%
in_customers = inflows['prism_consumer_id'].nunique()
out_customers = outflows['prism_consumer_id'].nunique()
print(f"Unique Customers in Inflows: {in_customers}")
print(f"Unique Customers in Outflows: {out_customers}")

# %% [markdown]
# ### Category Counts

# %%
inflows['category'].value_counts()

# %%
outflows['category'].value_counts()

# %% [markdown]
# ### Unique Memos Per Category

# %%
inflows[['category','memo']].groupby('category').nunique()

# %%
outflows[['category','memo']].groupby('category').nunique()

# %% [markdown]
# Only 9 categories with more than 1 unique memos

# %% [markdown]
# ## Categorization by Memo

# %%
merchant_cat = ['EDUCATION', 'FOOD_AND_BEVERAGES', 'GENERAL_MERCHANDISE', 'GROCERIES', 'MORTGAGE','OVERDRAFT', 'PETS', 'RENT', 'TRAVEL']
merchant_df = outflows[outflows['category'].isin(merchant_cat)].reset_index()
merchant_df

# %%
merchant_df.groupby(['category','memo']).count().sort_values(by=['category','index'], ascending=False).groupby('category').head(3)

# %%
inflows_consumers = inflows['prism_consumer_id'].unique()
in_train_users, in_test_users = train_test_split(inflows_consumers, test_size=0.2)#, random_state=42)
len(in_train_users), len(in_test_users)

# %%
in_train_df = inflows[inflows['prism_consumer_id'].isin(in_train_users)]
in_test_df = inflows[inflows['prism_consumer_id'].isin(in_test_users)]
len(in_train_df), len(in_test_df)

# %%
outflows_consumers = outflows['prism_consumer_id'].unique()
out_train_users, out_test_users = train_test_split(outflows_consumers, test_size=0.2, random_state=42)
len(out_train_users), len(out_test_users)

# %%
out_train_df = outflows[outflows['prism_consumer_id'].isin(out_train_users)]
out_test_df = outflows[outflows['prism_consumer_id'].isin(out_test_users)]
len(out_train_df), len(out_test_df)

# %% [markdown]
# ### Memo Cleaning

# %%
test_snippet = out_train_df['memo'].iloc[:20]
test_snippet

# %%
def cleaning(series):
    series = series.str.lower() # CASE NORMALIZATION
    series = series.apply(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')) # CONVERT INTO UTF-8 

    # REMOVE @ BUT EXTRACT THE EMAIL DOMAIN
    def extract_email_domain(text):
        def replacer(match):
            full_email = match.group(0)
            domain = full_email.split('@')[1]  # take part after '@'
            return domain
        return re.sub(r'[\w\.-]+@[\w\.-]+', replacer, text)
    series = series.apply(extract_email_domain)

    # REMOVE ALL . UNLESS IT'S IN A WEBSITE OR EMAIL DOMAIN (remove standalone dots - not part of word/num)
    series = series.str.replace(r'(?<!\w)\.(?!\w)', '', regex=True)

    # REMOVE ALL # AND CHARACTERS AFTER IT (likely numbers)
    series = series.str.replace(r'#.*', '', regex=True)

    # REMOVE ALL % (not useful)
    series = series.str.replace('%', '')

    # REMOVE * (usually follows TST, so we can just use that)
    series = series.str.replace('*', '')

    # REMOVE PARENTHESES
    series = series.str.replace(r'[()]', '', regex=True)

    # REMOVE / AND REPLACE WITH ' '- occurs most commonly in dates but doesn't provide much info to repaying loans
    series = series.str.replace(r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b', '', regex=True)
    series = series.str.replace('/', ' ', regex=True)

    # REPLACE & WITH 'AND' UNLESS IT APPEARS IN 'AT&T' OR 'H&M'
    def replace_ampersand(text):
        if 'h&m' not in text and 'at&t' not in text:
            return text.replace('&', 'and')
        return text
    series = series.apply(replace_ampersand)

    # REMOVE ALL APOSTROPHES
    series = series.str.replace("'", "")

    # REPLACE '_' WITH ' '
    series = series.str.replace('_', ' ')

    # REMOVE ALL $
    series = series.str.replace('$', '')

    # REMOVE ALL :
    series = series.str.replace(':', '')

    # remove standalone sequence of + digits surrounded by word boundaries
    # chose 4 because after removing dashes from dates, we end up with many unneeded 4-digit numbers
    series = series.replace(r'\b\d{4,}\b', '', regex=True) 

    # REMOVE XXXX IF THERE ARE 4+ Xs
    series = series.str.replace(r'x{4,}', '', regex=True)

    # REMOVE ALL DASHES UNLESS SURROUNDED BY ALPHANUMERIC TEXT (detects phone numbers, codes, some merchant names)
    series = series.str.replace(r'(?<!\w)-{1,}(?!\w)', '', regex=True)

    # REMOVE INVISIBLE/ZERO-WIDTH CHARACTERS
    # series = series.str.replace(r'[\u200B\u200C\u200D\u2060\uFEFF\u00A0\u180E]', ' ', regex=True)

    # SPECIAL CASE: DOORDASH
    series = series.str.replace(r'(?<=doordash)(?=[A-Za-z])', ' ', regex=True)

    # REMOVE REDUNDANT WHITESPACE
    series = series.replace(r'\s+', ' ', regex=True).str.strip()

    return series

# %%
cleaning(test_snippet).to_list()

# %%
# Test series
test_memos = pd.Series([
    "POS PURCHASE - STARBUCKS 04/25",               # normal purchase with dash and date
    "Payment to user@google.com",                   # email extraction
    "H&M Store #12345",                             # & and # handling
    "T.J.Maxx (0425)",                              # parentheses, dot handling
    "Transfer 1234-5678",                           # long digits and dash
    "ACH CREDIT PAYROLL 50%",                       # % removal
    "VENMO PAYMENT TO john.doe@venmo.com",          # email domain + dot
    "ATM WDL $200*",                                # asterisk removal
    "AT&T Bill Payment 09/23",                      # & preserved in AT&T
    "Online subscription: Netflix.com",             # dot in domain,
    " POS PURCHASE STARBUCKS 0425 ",
    "ACH CREDIT 12345678"
])

# %%
cleaning(test_memos).to_list()

# %%
out_train_df['cleaned_memos'] = cleaning(out_train_df['memo'])
out_train_df

# %%
out_test_df['cleaned_memos'] = cleaning(out_test_df['memo'])
out_test_df

# %% [markdown]
# ### Feature Creation

# %%
out_train_df_modified = out_train_df.copy()
out_test_df_modified = out_test_df.copy()

# %%
# from earlier: Keeping just the 9 categories with unique memos
merchant_cat = ['EDUCATION', 'FOOD_AND_BEVERAGES', 'GENERAL_MERCHANDISE', 'GROCERIES', 'MORTGAGE','OVERDRAFT', 'PETS', 'RENT', 'TRAVEL']

# %%
out_train_df_modified = out_train_df_modified[out_train_df_modified['category'].isin(merchant_cat)]
out_test_df_modified = out_test_df_modified[out_test_df_modified['category'].isin(merchant_cat)]

# %%
y_train = out_train_df_modified['category']
y_test = out_test_df_modified['category']

# %%
# Dealing with Numeric Attributes
for df in [out_train_df_modified, out_test_df_modified]:
    df['posted_date'] = pd.to_datetime(df['posted_date']) # convert to datetime format
    df['day_of_week'] = df['posted_date'].dt.dayofweek.astype(int)
    df['day_of_month'] = df['posted_date'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['whole_dollar'] = round(df['amount']).astype(int)

# %%
preprocessor = ColumnTransformer(
    transformers = [
        # TF-IDF vectorizer - returns sparse
        ('text', TfidfVectorizer(
        max_features=5000, 
        ngram_range = (1,2), #drop too common items
        ), 'cleaned_memos'), 

        # standard scaler - returns dense
        ('numeric', StandardScaler(), ['amount', 'day_of_week', 'is_weekend', 'whole_dollar'])

    ], sparse_threshold=0.5
)

# %%
out_train_tfidf = preprocessor.fit_transform(out_train_df_modified)
out_test_tfidf = preprocessor.transform(out_test_df_modified)

# %%
tfidf_sparsity = 1 - (out_train_tfidf.nnz / (out_train_tfidf.shape[0] * out_train_tfidf.shape[1]))
print(f"TF-IDF sparsity: {tfidf_sparsity:.2%}")
print(f"TF-IDF non-zero ratio: {1-tfidf_sparsity:.2%}")

# %% [markdown]
# ### Model Creation

# %%
# Baseline model - no hyperparameter tuning needed
start_time = time.time()
model = LogisticRegression(max_iter = 1000, random_state=42)
model.fit(out_train_tfidf, y_train)
end_time = time.time()
print(f"time to complete: {(end_time - start_time)/60:.2f} minutes")


# %%
y_pred = model.predict(out_test_tfidf)

# %%
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %%
# Advanced model
adv_model = RandomForestClassifier(random_state = 42, n_jobs = -1)

# %%
start_time = time.time()
adv_model.fit(out_train_tfidf, y_train)
end_time = time.time()
print(f"time to complete: {(end_time - start_time)/60:.2f} minutes")

# %%
y_pred_adv = adv_model.predict(out_test_tfidf)

# %%
print(f"Accuracy: {accuracy_score(y_test, y_pred_adv):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_adv))

# %%
labels_lr_rf = [
    "EDUCATION", "FOOD_AND_BEVERAGES", "GENERAL_MERCHANDISE",
    "GROCERIES", "MORTGAGE", "OVERDRAFT", "PETS", "RENT", "TRAVEL"
]

# Logistic Regression
precision_lr = [0.84, 0.90, 0.94, 0.97, 0.95, 0.99, 0.98, 0.87, 0.96]
recall_lr    = [0.46, 0.95, 0.93, 0.93, 0.77, 0.93, 0.82, 0.75, 0.88]
f1_lr        = [0.60, 0.93, 0.93, 0.95, 0.85, 0.96, 0.89, 0.81, 0.92]

# Random Forest
precision_rf = [0.65, 0.89, 0.92, 0.95, 0.93, 0.99, 0.93, 0.78, 0.94]
recall_rf    = [0.34, 0.93, 0.92, 0.91, 0.82, 0.93, 0.79, 0.73, 0.84]
f1_rf        = [0.45, 0.91, 0.92, 0.93, 0.87, 0.96, 0.86, 0.75, 0.89]


# %%
x = np.arange(len(labels_lr_rf))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, f1_lr, width, label="Logistic Regression", color='red')
plt.bar(x + width/2, f1_rf, width, label="Random Forest", color='blue')

plt.xticks(x, labels_lr_rf, rotation=45, ha="right")
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Score Comparison: Logistic Regression vs Random Forest")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# **Figure 1**: This plot compares the per-class F1 scores of Logistic Regression (red) and Random Forest (blue), showing that Logistic Regression generally performs better across most transaction categories.

# %%
diff = np.array(f1_rf) - np.array(f1_lr)

plt.figure(figsize=(10, 2))
sns.heatmap([diff], annot=True, cmap="coolwarm", center=0,
            xticklabels=labels_lr_rf, yticklabels=["RF - LR"])
plt.title("F1 Score Difference (Random Forest – Logistic Regression)")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Figure 2**: This heatmap shows the difference in F1 scores between Random Forest and Logistic Regression for each class, with positive values indicating where Random Forest outperforms Logistic Regression.

# %% [markdown]
# ### Sentence Encoder

# %%
# 2️⃣ Prepare your data;
x_train = out_train_df_modified['cleaned_memos'].to_list()
y_train = out_train_df_modified['category'].to_list()
x_test = out_test_df_modified['cleaned_memos'].to_list()
y_test = out_test_df_modified['category'].to_list()

# %%
# 3️⃣ Load SentenceTransformer model
start_time = time.time()
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight, fast
setup_time = time.time() - start_time

# 4️⃣ Convert text to embeddings
start_time = time.time()
x_train_embeds = embedder.encode(x_train, show_progress_bar=True, batch_size=256)
x_test_embeds = embedder.encode(x_test, show_progress_bar=True, batch_size=256)
embedding_time = time.time() - start_time

# 5️⃣ Train Logistic Regression classifier
start_time = time.time()
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x_train_embeds, y_train)
train_time = time.time() - start_time

# 6️⃣ Inference
start_time = time.time()
y_pred = model.predict(x_test_embeds)
inference_time = time.time() - start_time

# 7️⃣ Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Setup time: {setup_time:.3f} seconds")
print(f"Embedding time: {embedding_time:.3f} seconds")
print(f"Training time: {train_time:.3f} seconds")
print(f"Inference time: {inference_time:.3f} seconds")
print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))


# %%
# Create the encoder
le = LabelEncoder()

# Fit on the training labels and transform both train and test
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)  # if you have y_test

# Now train the model
from xgboost import XGBClassifier
import time

start_time = time.time()
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)
model.fit(x_train_embeds, y_train_encoded)
train_time = time.time() - start_time

print(f"✅ Model trained in {train_time:.2f} seconds")

y_pred_encoded = model.predict(x_test_embeds)
y_pred = le.inverse_transform(y_pred_encoded)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {accuracy:.4f}\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))


# %%
labels = [
    "EDUCATION", "FOOD_AND_BEVERAGES", "GENERAL_MERCHANDISE",
    "GROCERIES", "MORTGAGE", "OVERDRAFT", "PETS", "RENT", "TRAVEL"
]

# Logistic Regression metrics
precision_lr = [0.77, 0.84, 0.87, 0.89, 0.95, 0.99, 0.91, 0.61, 0.89]
recall_lr    = [0.53, 0.87, 0.88, 0.83, 0.86, 0.95, 0.84, 0.47, 0.79]
f1_lr        = [0.63, 0.86, 0.88, 0.86, 0.91, 0.97, 0.87, 0.53, 0.84]
support_lr   = [934, 87465, 96877, 44339, 295, 779, 1836, 672, 10115]


# %%
# XGBoost metrics
precision_xgb = [0.86, 0.85, 0.90, 0.96, 1.00, 0.99, 0.94, 0.93, 0.98]
recall_xgb    = [0.52, 0.93, 0.91, 0.83, 0.29, 0.93, 0.74, 0.31, 0.80]
f1_xgb        = [0.64, 0.89, 0.90, 0.89, 0.45, 0.96, 0.82, 0.46, 0.88]
support_xgb   = [934, 87465, 96877, 44339, 295, 779, 1836, 672, 10115]


# %%
def plot_f1(labels, f1, title, color="#4DB6AC"):  # teal color
    plt.figure(figsize=(10,4))
    plt.bar(labels, f1, color=color)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Example:
plot_f1(labels, f1_lr, "Logistic Regression — F1 Score by Class", color="#4DB6AC")
plot_f1(labels, f1_xgb, "XGBoost — F1 Score by Class", color="#FFB74D")  # orange for XGBoost


# %% [markdown]
# **Figure 3**: This plot shows the F1 score for each class, highlighting the per-class performance of the model.

# %%
def plot_prec_recall(labels, precision, recall, title, color_precision="#E57373", color_recall="#FFB74D"):
    """
    Plot precision and recall side by side for each class.
    
    color_precision: bar color for precision
    color_recall: bar color for recall
    """
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(12,5))
    plt.bar(x - width/2, precision, width, label="Precision", color=color_precision)
    plt.bar(x + width/2, recall, width, label="Recall", color=color_recall)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Examples:
plot_prec_recall(labels, precision_lr, recall_lr, "Logistic Regression — Precision vs Recall", color_precision="#4DB6AC", color_recall="#FFB74D")
plot_prec_recall(labels, precision_xgb, recall_xgb, "XGBoost — Precision vs Recall", color_precision="#4DB6AC", color_recall="#FFB74D")


# %% [markdown]
# **Figure 4**: This plot compares precision and recall for each class, showing how the model balances correctness versus completeness in predictions.

# %%
def plot_support(labels, support, title, color="#4DB6AC"):
    """
    Plot class support (number of samples) on a log scale with customizable bar color.
    
    color: bar color
    """
    plt.figure(figsize=(10,4))
    plt.bar(labels, support, color=color)
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.ylabel("Support (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Example usage
plot_support(labels, support_lr, "Logistic Regression — Class Support", color="#4DB6AC")
plot_support(labels, support_xgb, "XGBoost — Class Support", color="#FFB74D")



# %% [markdown]
# **Figure 5**: This plot displays the number of samples per class on a log scale, showing the distribution of transaction categories for each model.

# %% [markdown]
# ### BERT

# %%
x_train = out_train_df_modified['cleaned_memos'].to_list()
y_train = out_train_df_modified['category'].to_list()
x_test = out_test_df_modified['cleaned_memos'].to_list()
y_test = out_test_df_modified['category'].to_list()

# %%
# ──────────────────────────────────────────────────────
# 0️⃣ Setup
# ──────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ──────────────────────────────────────────────────────
# 1️⃣ Load FinBERT model + tokenizer
# ──────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModel.from_pretrained("ProsusAI/finbert").to(device)
model.eval()

# ──────────────────────────────────────────────────────
# 2️⃣ Embedding function (batched)
# ──────────────────────────────────────────────────────
def get_finbert_embeddings(text_list, batch_size=32, max_length=128):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # mean pooling across tokens
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
        embeddings.append(emb)

        torch.cuda.empty_cache()

    return np.concatenate(embeddings, axis=0)

# ──────────────────────────────────────────────────────
# 3️⃣ Subset data if needed (for memory/time control)
# ──────────────────────────────────────────────────────
# Suppose your full lists are x_train, y_train, x_test, y_test
subset_train_size = 50000  # you can increase if memory allows
subset_test_size = 10000

x_train_sub = x_train[:subset_train_size]
y_train_sub = y_train[:subset_train_size]
x_test_sub = x_test[:subset_test_size]
y_test_sub = y_test[:subset_test_size]

# ──────────────────────────────────────────────────────
# 4️⃣ Generate embeddings & time it
# ──────────────────────────────────────────────────────
print("\nGenerating embeddings for FinBERT ...")
start_time = time.time()
X_train_emb = get_finbert_embeddings(x_train_sub, batch_size=32, max_length=128)
X_test_emb = get_finbert_embeddings(x_test_sub, batch_size=32, max_length=128)
embed_time = time.time() - start_time
print(f"Embedding generation time: {embed_time/60:.2f} minutes\n")

# ──────────────────────────────────────────────────────
# 5️⃣ Train Logistic Regression & time it
# ──────────────────────────────────────────────────────
print("Training Logistic Regression on FinBERT embeddings ...")
model_finbert_lr = LogisticRegression(
    solver="saga",
    max_iter=300,
    tol=1e-3,
    n_jobs=8,
    random_state=42,
    verbose=0
)

start_time = time.time()
model_finbert_lr.fit(X_train_emb, y_train_sub)
train_time = time.time() - start_time
print(f"Training complete in {train_time/60:.2f} minutes\n")

# ──────────────────────────────────────────────────────
# 6️⃣ Evaluate & time inference
# ──────────────────────────────────────────────────────
print("Evaluating model ...")
start_time = time.time()
y_pred = model_finbert_lr.predict(X_test_emb)
infer_time = time.time() - start_time

print(f"Inference time: {infer_time:.2f} seconds")
print(f"Accuracy: {accuracy_score(y_test_sub, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test_sub, y_pred))




# %%
# ───────────────────────────────
# 1️⃣ Define labels
# ───────────────────────────────
finbert_labels = [
    "EDUCATION",
    "FOOD_AND_BEVERAGES",
    "GENERAL_MERCHANDISE",
    "GROCERIES",
    "MORTGAGE",
    "OVERDRAFT",
    "PETS",
    "RENT",
    "TRAVEL"
]

# ───────────────────────────────
# 2️⃣ Confusion matrix function
# ───────────────────────────────
def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=labels
    )
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='.2%', colorbar=False)
    
    # Remove grid lines and borders
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Per-Category Performance:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=labels,
        digits=4
    ))

# ───────────────────────────────
# 3️⃣ Validation set
# ───────────────────────────────
# Reuse your precomputed embeddings if possible
# X_train_emb = get_finbert_embeddings(x_train_sub, batch_size=32, max_length=128)
y_val_pred = model_finbert_lr.predict(X_train_emb)  # X_val_emb is your precomputed validation embeddings

plot_confusion_matrix(y_train_sub, y_val_pred, finbert_labels, "FinBERT Normalized Confusion Matrix - Validation Set")

# ───────────────────────────────
# 4️⃣ Test set
# ───────────────────────────────
# Generate embeddings for test set if not precomputed
X_test_emb = get_finbert_embeddings(x_test_sub, batch_size=32, max_length=128)
y_test_pred = model_finbert_lr.predict(X_test_emb)

plot_confusion_matrix(y_test_sub, y_test_pred, finbert_labels, "FinBERT Normalized Confusion Matrix - Test Set")


# %%
# -----------------------------------
# 0. Fix Tokenizer Parallelism Warning
# -----------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------
# 1. Timer
# -----------------------------------
start_time = time.time()

# -----------------------------------
# 2. Dataset Prep
# -----------------------------------
df = out_train_df_modified.copy()

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["category"])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["cleaned_memos"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
val_df = pd.DataFrame({"text": val_texts, "label": val_labels})

# -----------------------------------
# 3. Tokenization
# -----------------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=32
    )

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
val_dataset   = Dataset.from_pandas(val_df).map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------------------
# 4. Model
# -----------------------------------
num_labels = len(label_encoder.classes_)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

# Freeze all except last transformer block
for name, param in model.distilbert.named_parameters():
    if "transformer.layer.5" not in name:
        param.requires_grad = False

# -----------------------------------
# 5. Metrics
# -----------------------------------
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# -----------------------------------
# 6. TrainingArguments (updated)
# -----------------------------------
training_args = TrainingArguments(
    output_dir="./distilbert_results",
    eval_strategy="epoch",      # updated
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=200,
    report_to="none",
)

# -----------------------------------
# 7. Trainer (updated)
# -----------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,   # updated
    compute_metrics=compute_metrics
)

# -----------------------------------
# 8. Train
# -----------------------------------
print("🚀 Starting training...")
trainer.train()

# -----------------------------------
# 9. Evaluate
# -----------------------------------
metrics = trainer.evaluate()
print("✅ Evaluation:", metrics)

# -----------------------------------
# 10. Timer
# -----------------------------------
end_time = time.time()
print(f"⏱️ Total training time: {(end_time - start_time)/60:.2f} minutes")


# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# -----------------------------------
# Validation Set
# -----------------------------------
val_predictions = trainer.predict(val_dataset)
y_val_pred = np.argmax(val_predictions.predictions, axis=1)
y_val_true = val_labels.values  # or just val_labels if it's already numpy array

# Normalized confusion matrix
cm_val_normalized = confusion_matrix(y_val_true, y_val_pred, normalize='true')

# Plot without grid lines or spines
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_val_normalized,
    display_labels=label_encoder.classes_
)
disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='.2%')
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.title('DistilBERT Normalized Confusion Matrix - Validation Set', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Detailed per-category breakdown
print("\n📊 Per-Category Performance - Validation Set:")
print(classification_report(
    y_val_true,
    y_val_pred,
    target_names=label_encoder.classes_,
    digits=4
))

# -----------------------------------
# Test Set
# -----------------------------------
test_df = out_test_df_modified.copy()
test_df["text"] = test_df["cleaned_memos"]  # rename for tokenizer
test_df["label"] = label_encoder.transform(test_df["category"])  # encode labels

test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

test_predictions = trainer.predict(test_dataset)
y_test_pred = np.argmax(test_predictions.predictions, axis=1)
y_test_true = test_df["label"].values

# Normalized confusion matrix
cm_test_normalized = confusion_matrix(y_test_true, y_test_pred, normalize='true')

# Plot without grid lines or spines
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_test_normalized,
    display_labels=label_encoder.classes_
)
disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='.2%')
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.title('DistilBERT Normalized Confusion Matrix - Test Set', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Detailed per-category breakdown
print("\n📊 Per-Category Performance - Test Set:")
print(classification_report(
    y_test_true,
    y_test_pred,
    target_names=label_encoder.classes_,
    digits=4
))


# %% [markdown]
# ## Income Prediction

# %%
# Total and Average transactions per consumer
transact_per_cust = inflows.groupby('prism_consumer_id')['memo'].count()
mean_trans_per_cust = transact_per_cust.mean()
print(f'Average number of transactions per customer: {mean_trans_per_cust}')
transact_per_cust

# %%
transact_per_cust_fig = px.box(x=transact_per_cust)
transact_per_cust_fig.update_layout(xaxis_title='Number of Transactions per Customer')
transact_per_cust_fig.show()

# %%
# Total inflows by category - proportion of amounts
(inflows.groupby('category')['amount'].sum() / inflows['amount'].sum()).sort_values(ascending=False)

# %%
inflows['category'].unique()

# %% [markdown]
# ### What Counts as Income?

# %% [markdown]
# What counts as income? 
# 
# 1. **PAYCHECK** – Direct deposits from an employer for salary or wages.
# 
# 2. ~~**EXTERNAL_TRANSFER** – Money coming from accounts at other banks or financial institutions.~~
# 
# 3. ~~**MISCELLANEOUS** – Any income that doesn’t fit into other categories; small or irregular sources.~~
# 
# 4. **INVESTMENT_INCOME** – Returns from investments such as dividends, interest, or capital gains.
# 
# 5. ~~**TAX** – Refunds from tax authorities (like IRS or state tax refunds).~~
# 
# 6. ~~**DEPOSIT** – General deposits, could include cash deposits, checks, or other unspecified inflows.~~
# 
# 7. ~~**SELF_TRANSFER** – Moving money between your own accounts (e.g., from savings to checking).~~
# 
# 8. ~~**REFUND** – Refunds from merchants for returned purchases or overpayments.~~
# 
# 10. ~~**PAYCHECK_PLACEHOLDER** – Likely a system-generated placeholder for expected paychecks; may not be actual deposited funds.~~
# 
# 11. **INSURANCE** – Payments from insurance claims or benefits (e.g., health, life, or property insurance payouts).
# 
# 12. **OTHER_BENEFITS** – Miscellaneous benefits from employers or government (non-wage, non-tax-related).
# 
# 13. **UNEMPLOYMENT_BENEFITS** – Payments from government unemployment insurance programs.
# 
# 14. ~~**LOAN** – Funds received from loans or lines of credit.~~
# 
# 15. ~~**SMALL_DOLLAR_ADVANCE** – Short-term small cash advances, often from payday-type loans or similar products.~~

# %%
# Total inflows by category - summed amounts
inflows.groupby('category')['amount'].sum().sort_values(ascending=False).map("{:,}".format)

# %%
# Total inflows by category - number of transactions
inflows.groupby('category')['amount'].count().sort_values(ascending=False).map("{:,}".format)

# %%
inflows['posted_date'] = pd.to_datetime(inflows['posted_date'])
inflows['year_month'] = inflows['posted_date'].dt.to_period('M').dt.to_timestamp()
inflows

# %%
# Mean + STD intervals between inflows by consumer
inflows = in_train_df.sort_values(['prism_consumer_id','posted_date'])
inflows['posted_date'] = pd.to_datetime(inflows['posted_date'])
inflows['days_since_last'] = inflows.groupby('prism_consumer_id')['posted_date'].diff().dt.days
regularity_stats = inflows.groupby('prism_consumer_id')['days_since_last'].agg(['mean','std'])
regularity_stats

# %%
regularity_stats['mean'].mean()

# %%
iqr = regularity_stats['mean'].quantile(0.75) - regularity_stats['mean'].quantile(0.25)
high_end = regularity_stats['mean'].quantile(0.75) + 1.5 * iqr
low_end = regularity_stats['mean'].quantile(0.25) - 1.5 * iqr # negative number, everything will be above lower bound
regularity_stats['irregular_income'] = regularity_stats['mean'] > high_end
regularity_stats

# %%
# Number of unique inflow sources (by category)
num_income_sources = in_train_df.groupby('prism_consumer_id')['category'].nunique().rename('num_income_sources')

# Proportion of inflow from top source
top_source_prop = (
    in_train_df
    .groupby(['prism_consumer_id','category'])['amount'].sum()
    .groupby(level=0)
    .apply(lambda x: x.max()/x.sum())
    .rename('top_source_prop')
)
# top_source_prop
plt.hist(top_source_prop)

# %%
# changing inflows
inflows['category'] = inflows['category'].replace({
    'PAYCHECK_PLACEHOLDER': 'PAYCHECK'
})


# %% [markdown]
# ### Regularity Prediction

# %%
# amount consistency
inflows['amount_mean'] = inflows.groupby(['prism_consumer_id', 'category'])['amount'].transform('mean')
inflows['amount_std'] = inflows.groupby(['prism_consumer_id', 'category'])['amount'].transform('std')

inflows['regular_amount'] = abs(inflows['amount'] - inflows['amount_mean']) < inflows['amount_std']

inflows.head()

# %%
unique_amounts = inflows.groupby('category')['amount'].unique()
unique_amounts

# %%
regular_counts = inflows.groupby('category')['regular_amount'].value_counts().unstack(fill_value=0)
regular_counts


# %% [markdown]
# for categories with high `True` counts indiciate consistency in inflows. for example, `external_transfer`, `self_transfer`, `deposit` have high true counts which is pretty repetitive financial habits. but for `paycheck`, there is a larger variance of how much people make. 
# 
# we can use amount stability to help helps identify regular income streams even when pay schedules are irregular or when paycheck amounts fluctuate due to taxes, hourly work, or overtime. this provides an additional measure of income consistency beyond timing or periodicity.

# %%
# frequency consistency
freq = inflows[inflows['category'] == 'PAYCHECK'].groupby('prism_consumer_id').size()
freq

# %%
# weekday consistency
inflows['posted_date'] = pd.to_datetime(inflows['posted_date'], errors='coerce')

inflows['year'] = inflows['posted_date'].dt.year

paycheck_counts = (
    inflows[inflows['category']=='PAYCHECK']
    .groupby(['prism_consumer_id','year'])
    .size()
)
paycheck_counts

# %%
inflows['weekday'] = inflows['posted_date'].dt.dayofweek  # Monday=0

weekday_counts = (
    inflows[inflows['category']=='PAYCHECK']
    .groupby('weekday')
    .size()
)
weekday_counts


# %%
plt.figure(figsize=(8, 5))
plt.bar(weekday_counts.index, weekday_counts.values)
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.xlabel('Weekday')
plt.ylabel('Number of Paychecks')
plt.title('Distribution of Paychecks by Weekday')
plt.tight_layout()
plt.show()

# %%
weekday_score_map = {
    0: 0.1,  # Monday
    1: 0.2,  # Tuesday
    2: 0.4,  # Wednesday
    3: 0.9,  # Thursday
    4: 1.0,  # Friday
    5: 0.0,  # Saturday
    6: 0.0   # Sunday
}

inflows['weekday_score'] = inflows['weekday'].map(weekday_score_map)
inflows


# %%
dominant_weekday = (
    inflows[inflows['category']=='PAYCHECK']
    .groupby('prism_consumer_id')['weekday']
    .agg(lambda x: x.mode().iloc[0])
    .rename('dominant_payday_weekday')
)

inflows = inflows.merge(dominant_weekday, on='prism_consumer_id', how='left')


# %%
inflows['matches_person_weekday'] = (
    inflows['weekday'] == inflows['dominant_payday_weekday']
)
inflows['weekday_is_payday'] = (
      0.7 * inflows['weekday_score']
    + 0.3 * inflows['matches_person_weekday'].astype(float)
)
inflows = inflows.sort_values(['prism_consumer_id', 'category', 'posted_date'])

inflows['days_since_last'] = (
    inflows.groupby(['prism_consumer_id', 'category'])['posted_date']
      .diff()
      .dt.days
)
inflows.head()

# %%
spacing_stats = (
    inflows[inflows['category'] == 'PAYCHECK']
    .groupby('prism_consumer_id')['days_since_last']
    .median()
    .rename('median_gap')
)

inflows = inflows.merge(spacing_stats, on='prism_consumer_id', how='left')
inflows

# %%
def classify_pay_cycle(gap):
    if pd.isna(gap): 
        return np.nan
    if 10 <= gap <= 18:
        return 'biweekly'
    if 19 <= gap <= 25:
        return 'semimonthly'
    if 26 <= gap <= 35:
        return 'monthly'
    return 'irregular'

inflows['pay_cycle'] = inflows['median_gap'].apply(classify_pay_cycle)
inflows.head()

# %%
# source consistency
inflows['memo_clean'] = (
    inflows['memo']
    .str.upper()
    .str.replace(r'[^A-Z0-9 ]+', '', regex=True)
)

# Count most common memo for each user
employer_pattern = (
    inflows[inflows['category'] == 'PAYCHECK']
    .groupby('prism_consumer_id')['memo_clean']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    .rename('dominant_memo')
)

inflows = inflows.merge(employer_pattern, on='prism_consumer_id', how='left')

inflows['same_employer'] = inflows['memo_clean'] == inflows['dominant_memo']
inflows.head()

# %%
inflows['income_likelihood_score'] = (
      0.45 * inflows['regular_amount'].astype(int)
    + 0.15 * inflows['weekday_is_payday'].astype(int)
    + 0.25 * inflows['same_employer'].astype(int)
    + 0.15 * (inflows['pay_cycle'].isin(['biweekly','semimonthly','monthly']).astype(int))
)
inflows.head()

# %%
category_income_scores = (
    inflows.groupby('category')['income_likelihood_score']
    .mean()
    .sort_values(ascending=False)
)

category_income_scores


# %%
threshold = 0.5

income_flag_rate = (
    inflows.assign(high_score = inflows['income_likelihood_score'] > threshold)
           .groupby('category')['high_score']
           .mean()
           .sort_values(ascending=False)
)

income_flag_rate


# %%
import pandas as pd
from scipy.stats import ttest_ind

# Split into PAYCHECK and non-PAYCHECK
paycheck_scores = inflows[inflows["category"] == "PAYCHECK"]["income_likelihood_score"]
other_scores = inflows[inflows["category"] != "PAYCHECK"]["income_likelihood_score"]

# Basic stats
summary_stats = pd.DataFrame({
    "Group": ["PAYCHECK", "Other"],
    "Mean": [paycheck_scores.mean(), other_scores.mean()],
    "Median": [paycheck_scores.median(), other_scores.median()],
    "Std": [paycheck_scores.std(), other_scores.std()],
    "Count": [paycheck_scores.count(), other_scores.count()]
})

print(summary_stats)

# T-test
t_stat, p_value = ttest_ind(paycheck_scores, other_scores, equal_var=False)

print("\nT-test comparing PAYCHECK vs Other categories:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")




