
# Utilizing Cashflow Underwriting for Fairer Credit Assessment 
**Contributors:** Jasmine Hong, Heidi Tam, David Tsukamoto, Ellie Wang <br>
**Creation Date:** October 8, 2025 <br>
**Last Updated:** November 9, 2025 <br>

## Overview
Currently, one of the most widely used metrics for evaluating an individual's likelihood to pay back a loan is through the credit score, such as FICO or VantageScore. However, these metrics have their own limitations. Elderly people, for instance, may have not made purchases in the recent past, which can lower their credit score and make it more difficult for them to make large purchases, even if they previously maintained high credit scores and paid all their bills on time. On the other side of the spectrum, younger people may be reliable individuals but have a low credit score due to limited credit history. This project aims to use natural language processing to better understand the likelihood of people paying off their loans in two parts: <br> <br>

1) From October to December 2025, we use the text from banking transaction memos (about 2 million records) and build a strong and reliable model, aiming to classify the spending category each transaction memo falls under. Some example categories include education, food and beverages, and general merchandise. 

2) From January to March 2026, our mission is to use natural language processing to develop a more advanced machine learning model that provides a reliable score that estimates credit risk. 

## Running the Project
1) Navigate into the respective folder and run the following command in your command line or terminal: <br>
```git clone https://github.com/dtsukamoto/DSC180A-B01-2.git```

2) Set up the environment and activate it: <br>
``` conda env create -f environment.yml``` <br>
``` conda activate your-env-name```

3) If you have access to our data, create a folder called ```data``` in the root directory and place the ```ucsd-inflows.pqt``` and ```ucsd-outflows.pqt``` files into it. These represent the inflows (money that flows inward to one's bank account) and the outflows (people's spendings), respectively. 

4) Run the entire pipeline with: <br>
```python3 run.py```

## File Structure
```project-root/
│
├── README.md                               # Project overview and documentation
│
├── notebooks/                              # Weekly notebooks and progress checkpoints
│   ├── EDA_w2.ipynb                        # EDA
│   ├── Memo_Cleaning_w3.ipynb              # Text preprocessing and memo cleaning
│   ├── Feature_Creation_w4.ipynb           # Feature engineering and baseline models
│   ├── q1_checkpoint.ipynb                 # Quarter 1 checkpoint code
│   ├── q1_report.ipynb                     # Quarter 1 report code
│
├── environment.yml                         # Conda environment specification for reproducibility 
│
├── .gitignore                              # Git exclusion rules
│
└── run.py                                  # Main script for replicating all analysis and models                             
```

## Conclusion
In summary, our goal was to improve transaction classification to enable more inclusive credit evaluation, especially for individuals with limited credit history who may be disadvantaged under traditional scoring frameworks. After applying Regex preprocessing and testing several models, DistilBERT delivered the strongest performance, showing that efficient transformer models can classify transactions accurately without heavy compute. Our findings support the potential of cash-flow underwriting to broaden credit access and offer more comprehensive insight of financial behavior. While our results are promising, future work could incorporate additional structured features, explore improved fine-tuning strategies, and evaluate model performance on larger real-world datasets. 

Ultimately, this work demonstrates how modern NLP models can help financial institutions effectively evaluate consumers, improving access to financial products for underserved groups and providing a scalable path toward fairer, more inclusive credit decisioning.
