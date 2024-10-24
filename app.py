import streamlit as st
import speech_recognition as sr
import pandas as pd
import plotly.express as px # type: ignore
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import os
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator
from io import StringIO
# Path to your logo file

LOGO_PATH = r"C:\Users\msi\Desktop\data_science_instructor\output (1).jpg"


# Display the logo at the top of the app with a smaller size
st.image(LOGO_PATH)  # Adjust the width as needed

# Display the logo at the top of the app
st.logo(LOGO_PATH)


USER_CSV = r'C:\Users\msi\Desktop\data_science_instructor\USER.csv'
# Load user credentials from CSV
def load_users():
    if os.path.exists(USER_CSV):
        return pd.read_csv(USER_CSV)
    else:
        return pd.DataFrame(columns=["username", "password"])

# Save user credentials to CSV
def save_users(users_df):
    users_df.to_csv(USER_CSV, index=False)

# Function to handle user registration
def register_user(username, password):
    users_df = load_users()
    if username in users_df['username'].values:
        return False
    users_df = users_df.append({"username": username, "password": password}, ignore_index=True)
    save_users(users_df)
    return True

# Function to handle user login
def login_user(username, password):
    users_df = load_users()
    user = users_df[users_df['username'] == username]
    if not user.empty and user.iloc[0]['password'] == password:
        return True
    return False
b=0
# Function to display the login form
def display_login():
    st.header("Login :")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Login successful!")
        
            
        else:
            st.error("Invalid username or password")
    # Documentation and Help Section
    st.title("📚 Documentation and Help")

    # User Guides
    st.header("User Guides:")
    st.write("""
    Welcome to the user guide for PreXplore! This section provides detailed information on how to use the application effectively.
    
    1. **Login and Registration**:
       - Users can log in using their username and password. If you're new, please register first.

    2. **Data Upload**:
       - Users can upload datasets in CSV, Excel, or JSON formats. Use the file uploader to select your file.

    3. **Data Exploration**:
       - After uploading, you can view the dataset, check for null values, and perform data cleaning operations such as removing duplicates and imputing missing values.

    4. **Data Analysis**:
       - Explore various analytical features like generating correlation matrices, visualizations (pie charts, histograms), and data sampling.

    5. **Feedback**:
       - Users can provide feedback on their experience using the application, which helps improve future versions.
    """)

    # FAQs
    st.header("Frequently Asked Questions (FAQs):")
    faq_list = [
        ("Q1: How do I reset my password?", "A1: Currently, the application does not support password reset. Please contact support for assistance."),
        ("Q2: What file formats are supported?", "A2: The application supports CSV, Excel, and JSON file formats for data uploads."),
        ("Q3: How can I provide feedback?", "A3: You can provide feedback using the Feedback section in the application after you have logged in."),
        ("Q4: Who can I contact for support?", "A5: You can contact support via email at lassouedghazi21@gmail.com.")
    ]

    for question, answer in faq_list:
        st.write(question)
        st.write(answer)
    # Developer Information Section
    st.title("👨‍💻 Developer Information")

    st.write("This application was developed by Ghazi Lassoued. Here are my contact details:")

# Displaying contact information
    st.write("**LinkedIn:** [Ghazi Lassoued](https://www.linkedin.com/in/ghazi-lassoued-983419239/)")
    st.write("**Email:** [lassouedghazi21@gmail.com](mailto:lassouedghazi21@gmail.com)")
    st.write("**Phone Number:** +216 95292668")


    

# Function to display the registration form
def display_registration():
    st.header("Register :")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    if st.button("Register"):
        if register_user(username, password):
            st.success("Registration successful! You can now log in.")
        else:
            st.error("Username already taken. Please choose a different one.")

# Add custom CSS for styling
st.markdown(
    """
    <style>
        /* Set the background color and text color */
        body {
            background-color: #f5f5f5;
            color: #333;
        }
        /* Style the title */
        h1 {
            color: #00FFFF;
            font-size: 5em;
            text-align: center;
        }
        /* Style subheaders */
        h2 {
            color: #f0078f;
            font-size: 2em;
        }
        /* Style buttons */
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        /* Style markdown text */
        .result {
            font-size: 1.2em;
            margin: 10px 0;
            padding: 10px;
            background-color: #4CAF50;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        /* Style text input and text area */
        .stTextInput, .stTextArea {
            background-color: #4CAF50;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }
        /* Style for the dataframe preview */
        .stDataFrame {
            border-radius: 5px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Check if the user is logged in
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Display login or registration forms based on user session
if not st.session_state['logged_in']:
    option = st.sidebar.selectbox("Login or Register", ["Login", "Register"])
    if option == "Login":
        display_login()
    elif option == "Register":
        display_registration()
else:
    st.sidebar.success(f" 🔓 Logged in as {st.session_state['username']}")
    b=1
    





# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the data preprocessing guide from text file
with open('data_preprocessing_guide.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

# Tokenize the text into sentences
sentences = sent_tokenize(text_data)

# Preprocess the data
lemmatizer = WordNetLemmatizer()
translator = Translator()

def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word.isalpha()]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)
def save_feedback(username, feedback, rating):
    # Create a dictionary for the feedback data
    feedback_data = {
        "Username": [username],
        "Rating": [rating],
        "Feedback": [feedback],
        "Timestamp": [pd.Timestamp.now()]
    }
    
    # Create a DataFrame from the feedback data
    feedback_df = pd.DataFrame(feedback_data)
    
    # Check if the feedback.csv file exists
    if not os.path.exists("feedback.csv"):
        # If not, create the CSV file and save the DataFrame
        feedback_df.to_csv("feedback.csv", index=False)
    else:
        # If it exists, append the DataFrame to the existing CSV file
        feedback_df.to_csv("feedback.csv", mode='a', header=False, index=False)

# Function to find synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Function to expand the query with synonyms
def expand_query(query):
    words = query.split()
    expanded_words = []
    for word in words:
        expanded_words.append(word)
        synonyms = get_synonyms(word)
        expanded_words.extend(synonyms)
    return " ".join(set(expanded_words))

# Function to find the most relevant sentences using TF-IDF and cosine similarity
@st.cache_data
def get_most_relevant_sentences(query, num_responses=3):
    query_processed = preprocess(query)
    expanded_query = expand_query(query_processed)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    sentences_combined = sentences + [expanded_query]

    # Transform the sentences into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(sentences_combined)

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Get indices of the most relevant sentences
    most_relevant_indices = cosine_similarities[0].argsort()[-num_responses:][::-1]

    return [sentences[i] for i in most_relevant_indices]

# Speech Recognition
LANGUAGES = {
    'English': 'en',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Chinese (Mandarin)': 'zh-cn',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Arabic': 'ar',
    'Hindi': 'hi',
}

def transcribe_speech(api_choice, lang):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio_text = r.listen(source)
        st.info("📝 Transcribing...")

        try:
            if api_choice == "Google":
                text = r.recognize_google(audio_text, language=LANGUAGES[lang])
            elif api_choice == "Sphinx":
                text = r.recognize_sphinx(audio_text)
            else:
                text = "Invalid API choice"
            return text
        except sr.UnknownValueError:
            return "⚠️ Sorry, I did not understand that."
        except sr.RequestError as e:
            return f"⚠️ Could not request results; {e}"

def translate_text(text, dest_lang):
    translation = translator.translate(text, dest=dest_lang)
    return translation.text



def main():
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''
        st.success("You have been logged out.")
    
    a=0
    st.title("PreXplore 🌟")
    st.markdown(
    """
    ### Welcome to PreXplore! 🎉

    **Unlock the Secrets of Your Data!** 🔑

    At **PreXplore**, we empower you to transform your raw data into insightful narratives. Whether you're a data novice or a seasoned analyst, our application is designed to make your data preprocessing and exploratory data analysis (EDA) journey smooth and intuitive. 

    🌟 **What You Can Do:**
    
    - **Ask Questions**: Interact with our friendly chatbot—simply type or speak your queries about data preprocessing and get instant answers!
    
    - **Explore Your Data**: Discover hidden patterns and valuable insights through engaging EDA techniques that bring your data to life.
    
    - **Preprocess with Ease**: Simplify your workflow with easy-to-use buttons for common data preprocessing tasks—no coding required!

    Ready to embark on a data adventure? Let’s dive into the world of **PreXplore** and unleash the potential of your data! 🚀
    """
)

    st.markdown(""" 
    ## 🤖 How to Use the Chatbot
    Here are some example questions you can ask:
    - **Data Cleaning:**
      - "How can I handle missing values in my dataset?"
      - "What are the common techniques for dealing with outliers?"
    - **Data Transformation:**
      - "How do I normalize my data?"
      - "What are the different ways to encode categorical variables?"
    - **Feature Engineering:**
      - "Can you explain feature selection methods?"
      - "How do I create new features from existing data?"
    - **Exploratory Data Analysis:**
      - "What are some visualizations I can use to understand my data?"
      - "How do I identify correlations between variables?"
    """)

    api_choice = st.selectbox("Choose API", ["Google", "Sphinx"])
    lang_choice = st.selectbox("Choose Language", list(LANGUAGES.keys()))
    lang = lang_choice

    user_input = st.text_input("Type your question about data preprocessing:")

    if st.button("🎤 Record Speech"):
        speech_input = transcribe_speech(api_choice, lang)
        st.write("You said: ", speech_input)
        user_input = speech_input

    if user_input:
        with st.spinner('Fetching answer...'):
            try:
                responses = get_most_relevant_sentences(user_input)
                if responses and any("data preprocessing" in response.lower() for response in responses):
                    translated_responses = [translate_text(response, LANGUAGES[lang]) for response in responses]

                    st.subheader('📊 Responses:')
                    for i, response in enumerate(translated_responses, start=1):
                        st.markdown(f'<p class="result">{i}. {response}</p>', unsafe_allow_html=True)
                else:
                    st.warning("Sorry, I couldn't find any relevant information for your query. Please try asking something else related to data preprocessing.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

        feedback = st.radio("Was this answer helpful?", ('Select an option', 'Yes', 'No'))
        if feedback != 'Select an option':
            st.success(f'Thank you for your feedback: {feedback}!')

    st.title("📊 Exploratory Data Analysis & Data Preprocessing")
    
    # Select File Format
    file_format = st.selectbox("Select File Format to Import", options=['CSV', 'Excel', 'JSON'])

# File Uploader
    if file_format == 'CSV':
     uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    elif file_format == 'Excel':
     uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls'])
    elif file_format == 'JSON':
     uploaded_file = st.file_uploader("Upload your JSON file", type=['json'])
    

    if uploaded_file is not None:
        # Read data based on selected format
       if file_format == 'CSV':
        df = pd.read_csv(uploaded_file)
        a=1
       elif file_format == 'Excel':
        df = pd.read_excel(uploaded_file)
        a=1
       elif file_format == 'JSON':
        df = pd.read_json(uploaded_file)
        a=1
       

       
    if a==1:
        st.write(df.head())
        st.write("### 📋 Show Dataset Info")
        st.write("This button provides a summary of the DataFrame, including the number of non-null entries and data types.")
        if st.button("Show Dataset Info"):
            buffer = StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text_area("Dataset Info", value=s, height=300)

        st.write("### 📊 Describe Dataset")
        st.write("This button shows basic statistical details of the DataFrame, such as count, mean, and standard deviation.")
        if st.button("Describe Dataset"):
            st.dataframe(df.describe())
        
        st.write("### 🔍 Check for Null Values")
        st.write("This button will check for null values in each column of your dataset.")
        if st.button("Check for Null Values"):
            st.write("Null values in the dataset:")
            st.write(df.isnull().sum())
        

        st.write("### 📉 Show Percentage of Null Values")
        st.write("This button displays the percentage of null values in each column of the dataset.")
        if st.button("Show Percentage of Null Values"):
           null_percentage = df.isnull().mean() * 100
           st.write(null_percentage)
        # Delete columns with 90%+ null values
        st.write("### ❌ Delete Columns with High Null Values")
        st.write("This button deletes columns that have 90% or more null values.")
        if st.button("Delete Columns with 90%+ Null Values"):
         null_percentage = df.isnull().mean() * 100
         cols_to_delete = null_percentage[null_percentage >= 90].index.tolist()  # Convert to list for better display
         if len(cols_to_delete) > 0:
            df.drop(columns=cols_to_delete, inplace=True)
            st.success(f"Deleted columns: {cols_to_delete}")
            cleaned_csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned CSV without Columns >90% Null Values",
            data=cleaned_csv,
            file_name="cleaned_data_no_columns_90_percent_null.csv",
            mime='text/csv')
         else:
            st.warning("No columns with 90% or more null values found.")
        st.warning(""" ⚠️
Deleting columns with 90% or more null values can lead to a loss of important data.
 Use this option only when the remaining columns provide sufficient information for your analysis.
 Avoid using this option if the columns contain critical data or if alternative methods, 
such as imputation, can handle the missing values effectively.""")

        

        
        st.write("### 🗑️ Remove Null Values and Download")
        st.write("This button will remove all null rows from the dataset and allow you to download the cleaned dataset.")
        if st.button("Remove Null Values and Download"):
            cleaned_df = df.dropna()
            st.write("Rows with null values have been removed.")
            st.write(cleaned_df.head())
            csv = cleaned_df.to_csv(index=False)
            st.download_button(label="📥 Download Cleaned CSV without Null Values", data=csv, file_name="cleaned_data_no_null_values.csv", mime='text/csv')
        st.warning(""" ⚠️
Removing rows with null values will delete all records that contain any missing data. 
Use this option carefully as it can result in significant data loss, especially if many rows have at least one null value. 
Ensure that the dataset still retains enough data for meaningful analysis before proceeding with this option. 
Consider other methods, such as imputation, if preserving as much data as possible is important.""")
        
        
        # Button to impute null values in numerical columns with mean
        st.write("### 🔢 Impute (Replace) Null Values In Numerical Columns With Mean")
        st.write("This button will impute (replace) null values in numerical columns with mean and allow you to download the cleaned dataset.")
        if st.button("Impute Null Values with Mean"):
            def impute_nulls_with_mean(df):
                num_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in num_cols:
                    mean_value = df[col].mean()
                    df[col].fillna(mean_value, inplace=True)
                return df
            imputed_data = impute_nulls_with_mean(df)
            st.write("Data after imputing nulls with mean:")
            st.dataframe(imputed_data)
            csv_imputed_mean = imputed_data.to_csv(index=False)
            st.download_button("Download Imputed Data", csv_imputed_mean, "mean_imputed_data.csv", "text/csv")
        # Detailed warning text with conditions
        st.warning( "⚠️ "
    "Imputing numerical null values with the mean can lead to bias in your data. "
    "This is because the mean is sensitive to extreme values (outliers), "
    "which can distort the distribution of your data. "
    "Consider using this method only when:\n"
    "- The data is normally distributed.\n"
    "- There are no significant outliers affecting the mean.\n\n"
    "In other cases, consider using alternative imputation methods such as median or mode.\n"
    
)


        
        st.write("### 🔢 Impute (Replace) Null Values In Numerical Columns With Median")
        st.write("This button will impute (replace) null values in numerical columns with median and allow you to download the cleaned dataset.")
        if st.button("Impute Null Values with Median"):
            def impute_nulls_with_median(df):
                num_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in num_cols:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
                return df
            imputed_data = impute_nulls_with_median(df)
            st.write("Data after imputing nulls with median:")
            st.dataframe(imputed_data)
            csv_imputed_median = imputed_data.to_csv(index=False)
            st.download_button("Download Imputed Data", csv_imputed_median, "median_imputed_data.csv", "text/csv")
        st.warning(""" ⚠️ Imputing numerical null values with the median is generally more robust than using the mean, especially when dealing with outliers.
    However, always consider the distribution of your data before deciding on the imputation method.""")
      

        st.write("### 🔢 Impute (Replace) Null Values In Categorical Columns With Mode")
        st.write("This button will impute (replace) null values in Categorical columns with the mode and allow you to download the cleaned dataset.")
        if st.button("Impute Null Values with Mode"):
            def impute_nulls_with_mode(df):
                num_cols = df.select_dtypes(include=['object']).columns
                for col in num_cols:
                    mode_value = df[col].mode()[0]
                    df[col].fillna(mode_value, inplace=True)
                return df
            imputed_data = impute_nulls_with_mode(df)
            st.write("Data after imputing nulls with mode:")
            st.dataframe(imputed_data)
            csv_imputed_mode = imputed_data.to_csv(index=False)
            st.download_button("Download Imputed Data", csv_imputed_mode, "mode_imputed_data.csv", "text/csv")
        st.warning(""" ⚠️ Imputing categorical null values with the mode can be useful when you want to replace missing values with the most common value.
    However, consider the context of your data before deciding on the imputation method.""")
        
















        
        st.write("### 🔄 Check for Duplicate Values")
        st.write("This button will check for duplicate values in your dataset.")
        if st.button("Check for Duplicate Values"):
            st.write("Duplicate values in the dataset:")
            st.write(df.duplicated().sum())

        st.write("### 🗑️ Remove Duplicate Values and Download")
        st.write("This button will remove all duplicate rows from the dataset and allow you to download the cleaned dataset.")
        if st.button("Remove Duplicate Values and Download"):
            cleaned_df = df.drop_duplicates()
            st.write("Duplicate rows have been removed.")
            st.write(cleaned_df.head())
            csv = cleaned_df.to_csv(index=False)
            st.download_button(label="📥 Download Cleaned CSV without Duplicates", data=csv, file_name="cleaned_data_no_duplicates.csv", mime='text/csv')
        st.warning(""" ⚠️ Removing duplicate rows will delete all records that are identical across all columns. 
Use this option carefully, as it can result in the loss of data that might be important if duplicates are intentional or necessary. 
Ensure that the removal of duplicates does not adversely affect the integrity or completeness of the dataset before proceeding.""")
        # Button to check for outliers
        
        st.write("### 🔍 Identify Outliers In Each Numeric Column Of Your Dataset Using The IQR Method")

        def detect_outliers(df):
          outliers = {}
          for col in df.select_dtypes(include=['float64', 'int64']):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
          return outliers

        if st.button("Check for Outliers"):
          outliers_dict = detect_outliers(df)
          total_outliers = sum(outliers_dict.values())
    
          if total_outliers == 0:
           st.write("No outliers detected in the dataset.")
          else:
           st.write("Outliers detected in the dataset:")
           for col, count in outliers_dict.items():
            st.write(f"{col}: {count} outliers")
         # Button to remove outliers
        st.write("### 🗑️ Remove outliers From Each Numeric Column Of Your Dataset Using The IQR Method And Download")
        st.write("This button will remove outliers from each numeric column of your dataset using the IQR method and allow you to download the cleaned dataset.")
        if st.button(" Remove Outliers and Download"):
            st.warning(""" ⚠️ Removing outliers using the IQR method will eliminate data points that fall outside the range defined by 1.5 times the interquartile range (IQR) from the first quartile (Q1) and third quartile (Q3) for each numeric column. 
                      This can significantly alter your dataset, especially if it contains valid extreme values that are not true outliers. 
                       Ensure you understand the impact of removing these data points on your analysis before proceeding.""")
            def remove_outliers(df):
                df_cleaned = df.copy()
                for col in df.select_dtypes(include=['float64', 'int64']):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                return df_cleaned
        

            cleaned_df = remove_outliers(df)
            st.write("Outliers have been removed.")
            st.dataframe(cleaned_df.head())
            csv = cleaned_df.to_csv(index=False)
            st.download_button(label="📥 Download Cleaned CSV without Outliers", data=csv, file_name="cleaned_data_no_outliers.csv", mime='text/csv')
        st.warning(""" ⚠️ Removing outliers using the IQR method will eliminate data points that fall outside the range defined by 1.5 times the interquartile range (IQR) from the first quartile (Q1) and third quartile (Q3) for each numeric column. 
                      This can significantly alter your dataset, especially if it contains valid extreme values that are not true outliers. 
                       Ensure you understand the impact of removing these data points on your analysis before proceeding.""")


        st.write("### 🔍 Show Unique Values")
        st.write("This button provides the unique values for each categorical column in the dataset.")
        if st.button("Show Unique Values"):
         unique_values = {col: df[col].unique() for col in df.select_dtypes(include=['object', 'category']).columns}
         for col, values in unique_values.items():
          st.write(f"**{col}**: {values}")
        
        if 'selected_var' not in st.session_state:
            st.session_state.selected_var = None
        if st.button("Choose a Categorical Variable To Visualize "):
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(categorical_columns) == 0:
                st.warning("No categorical variables found in the dataset.")
            else:
                st.session_state.selected_var = st.selectbox("Select a Categorical Variable", categorical_columns)
        if st.session_state.selected_var is not None:
            if st.button("Show Pie Chart"):
                selected_variable = st.session_state.selected_var
                st.write(f"Pie Chart for {selected_variable}")
                # Prepare data for pie chart
                pie_data = df[selected_variable].value_counts().reset_index()
                pie_data.columns = [selected_variable, 'count']
                total_count = pie_data['count'].sum()
                threshold = 0.05  # Minimum percentage threshold (e.g., 5%)
                pie_data['percentage'] = pie_data['count'] / total_count
                pie_data = pie_data[pie_data['percentage'] >= threshold]  # Keep only categories above threshold
                other_count = df[selected_variable].value_counts()[df[selected_variable].value_counts() < total_count * threshold].sum()
                if other_count > 0:
                      pie_data = pie_data.append({selected_variable: 'Other', 'count': other_count, 'percentage': other_count / total_count}, ignore_index=True)
                # Create and display the pie chart
                fig = px.pie(pie_data, names=selected_variable, values='count', title=f'Pie Chart of {selected_variable}')
                st.plotly_chart(fig)
        st.write("### ⚖️ Standardize Data And Download")
        st.write('''This button will  standardize the numeric columns of your dataset using z-score normalization 
    (subtracting the mean and dividing by the standard deviation for each numeric column and allow you to download the standardized dataset)''')
        if st.button("Standardize Data"):
            st.write("Standardizing the numeric columns of the dataset...")
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            scaler = StandardScaler()
            df_standardized = df.copy()
            df_standardized[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            st.write("Data has been standardized.")
            st.dataframe(df_standardized.head())
            csv = df_standardized.to_csv(index=False)
            st.download_button(label="📥 Download Standardized Data CSV", data=csv, file_name="standardized_data.csv", mime='text/csv')
        st.warning("""⚠️ Standardizing the data will transform all numeric columns to have a mean of 0 and a standard deviation of 1.
                      This process can be useful for algorithms that are sensitive to the scale of the data, but it may also
                      alter the interpretation of the data. Ensure you understand the impact of standardization on your analysis before proceeding.""")
        st.write("### 🔍 Generate Correlation Matrix")
        st.write("This button will generate and display a correlation matrix of the numerical variables in your dataset.")
        if st.button(" Generate Correlation Matrix"):
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            correlation_matrix = df[numeric_cols].corr()
            fig = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=list(correlation_matrix.columns),
            y=list(correlation_matrix.index),
            annotation_text=correlation_matrix.round(2).values,
            colorscale='Viridis'
            )
            st.plotly_chart(fig)
        st.warning("""⚠️ The correlation matrix will show the linear relationship between numerical variables in your dataset. 
                      Values close to 1 or -1 indicate strong correlations, while values close to 0 indicate weak correlations.""")
        st.write("### 🌡️ Generate Correlation Matrix Heatmap")
        st.write("This button will display a heatmap of the correlation matrix for the numerical variables in your dataset.")
        if st.button("Generate Heatmap"):
              corr_matrix = df.corr()
              fig, ax = plt.subplots()
              sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
              st.write("Correlation Matrix Heatmap:")
              st.pyplot(fig)
        st.warning("""
⚠️ A heatmap is a data visualization technique that uses color to represent the values in a matrix. It provides a clear visual representation of the correlations between numerical variables in your dataset, making it easier to identify patterns and relationships at a glance. 

Heatmaps are especially useful for quickly spotting strong correlations, positive or negative, among variables, allowing for better insights and informed decision-making. 
""")


        # Initialize session state variables if they do not exist
       # Initialize session state variables for filtering
        if 'selected_column' not in st.session_state:
         st.session_state.selected_column = None

        if 'selected_values' not in st.session_state:
         st.session_state.selected_values = []

        st.write("### 🗂️ Filter Categorical Data")
        st.write("This allows you to filter the dataset based on categorical variables and download the filtered dataset.")

        # Select a categorical column to filter
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        st.session_state.selected_column = st.selectbox("Select a categorical column", categorical_columns)

        if st.session_state.selected_column:
       # Unique values for the selected categorical column
         unique_values = df[st.session_state.selected_column].unique()
         st.session_state.selected_values = st.multiselect(
        f"Select values to filter {st.session_state.selected_column} by",
        unique_values,
        default=st.session_state.selected_values
    )

         if st.session_state.selected_values:
        # Filter the DataFrame based on the selected values
            filtered_df = df[df[st.session_state.selected_column].isin(st.session_state.selected_values)]
            st.write(f"Data filtered by {st.session_state.selected_column} with values {st.session_state.selected_values}:")
            st.dataframe(filtered_df)

        # Download button for filtered DataFrame
            csv = filtered_df.to_csv(index=False)
            st.download_button(label="📥 Download Filtered Data CSV", data=csv, file_name="filtered_data.csv", mime='text/csv')
         else:
           st.warning("Please select at least one value to filter by.")
        # Initialize session state for numerical filtering
        if 'numerical_column' not in st.session_state:
          st.session_state.numerical_column = None

        if 'min_value' not in st.session_state:
          st.session_state.min_value = 0.0

        if 'max_value' not in st.session_state:
            st.session_state.max_value = 0.0

        st.write("### 📊 Filter Numerical Data")
        st.write("This allows you to filter the dataset based on numerical variables and download the filtered dataset.")

# Select a numerical column to filter
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        st.session_state.numerical_column = st.selectbox("Select a numerical column", numerical_columns)

        if st.session_state.numerical_column:
    # Get the min and max values of the selected numerical column for the slider
         min_value = float(df[st.session_state.numerical_column].min())
         max_value = float(df[st.session_state.numerical_column].max())
    
    # Update session state for slider values
        st.session_state.min_value, st.session_state.max_value = st.slider(
        f"Select range for {st.session_state.numerical_column}",
        min_value=min_value,
        max_value=max_value,
        value=(min_value, max_value)
    )
    
    # Filter the DataFrame based on the selected range
        filtered_df = df[(df[st.session_state.numerical_column] >= st.session_state.min_value) &
                     (df[st.session_state.numerical_column] <= st.session_state.max_value)]
    
        st.write(f"Filtered data for {st.session_state.numerical_column} between {st.session_state.min_value} and {st.session_state.max_value}:")
        st.dataframe(filtered_df)

    # Download button for filtered DataFrame
        csv = filtered_df.to_csv(index=False)
        st.download_button(label="📥 Download Filtered Data CSV", data=csv, file_name="filtered_numerical_data.csv", mime='text/csv')
        
        

# Initialize session state for sample size and sample DataFrame
        if 'sample_size' not in st.session_state:
         st.session_state.sample_size = 10  # Default sample size
        if 'sample_df' not in st.session_state:
         st.session_state.sample_df = pd.DataFrame()  # Placeholder for sampled DataFrame

        st.write("### 🎲 Data Sampling")
        st.write("This button generates a random sample of the dataset.")

# Slider to select sample size, default value managed by session state
        st.session_state.sample_size = st.slider(
         "Select sample size", 
         min_value=1, 
         max_value=len(df), 
         value=st.session_state.sample_size
          )

        if st.button("Generate Sample"):
        # Generate the random sample and store it in session state
         st.session_state.sample_df = df.sample(n=st.session_state.sample_size)
         st.write(f"Random sample of {st.session_state.sample_size} rows:")
         st.dataframe(st.session_state.sample_df)

    # Optionally, add a download button for the sampled data
        csv = st.session_state.sample_df.to_csv(index=False)
        st.download_button(label="📥 Download Sampled Data CSV", data=csv, file_name="sampled_data.csv", mime='text/csv')

        # Feedback Section
    st.title("🙋🏻‍♂️ Feedback")

# Retrieve the username from the session state
    username = st.session_state.get('username', None)

# Text area for feedback
    feedback_text = st.text_area("Please provide your feedback here:")

# Radio buttons for rating
    rating = st.radio("Rate your experience:", [1, 2, 3, 4, 5], index=2)  # Default to 3

    if st.button("Submit Feedback"):
     if username and feedback_text:
        save_feedback(username, feedback_text, rating)  # Call the function to save feedback
        st.success("Thank you for your feedback!")
     else:
        st.warning("Please provide your feedback before submitting.")



        




        



       


if __name__ == "__main__" and b==1:
    main()
