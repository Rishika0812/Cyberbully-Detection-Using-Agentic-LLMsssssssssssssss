import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from PIL import Image

# Set custom page config (MUST BE FIRST)
st.set_page_config(
    page_title="Cyberbully Detection Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Load preprocessed dataset
@st.cache_data
def load_data():
    return pd.read_csv('preprocessed_data.csv')  # Replace with your actual path

df = load_data()

# Add a gaming-style background and font
page_bg = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

[data-testid="stAppViewContainer"] {
    background-color: #1e1e1e;
    background-size: cover;
    color: #ffffff;
    font-family: 'Press Start 2P', cursive;
}
[data-testid="stSidebar"] {
    background-color: #242424;
}
h1, h2, h3, h4, h5, h6, p {
    color: #ffffff;
    text-shadow: 1px 1px 2px #000000;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation 🕹️")
st.sidebar.image("https://media2.giphy.com/media/HZrx8kjIA7lyeTqXVM/200.webp?cid=790b7611fttqwz07p1rfnr6ez90p0uwr301qpogu05s86rrm&ep=v1_gifs_search&rid=200.webp&ct=g", width=200)
pages = st.sidebar.radio(
    "Go to", 
    ["🏠 Home", "📊 EDA", "💬 Word Analysis", "📈 Advanced Visualizations", "🌌 Clustering"]
)

# Home Page
if pages == "🏠 Home":
    st.title("Cyberbully Detection Dashboard 🎮")
    st.markdown(
        """
        ## Welcome to the Cyberbully Detection Dashboard!
        This dashboard provides in-depth insights into cyberbullying trends across social media platforms.

        ### What to Expect:
        - **Exploratory Data Analysis (EDA)**: Analyze patterns and trends in cyberbullying data.
        - **Word Analysis**: Visualize common words and phrases in cyberbullying cases using word clouds.
        - **Advanced Visualizations**:
            - **TF-IDF**: Analyze term importance and frequency.
            - **PCA and t-SNE**: Dimensionality reduction techniques for visualizing data clusters.
        - **Interactive Clustering Visualizations**: Engage with data through clustering methods for deeper analysis.

        **Navigate through the sidebar to explore more!**
        """
    )
    
    # Add a GIF for visual appeal
    st.image("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExYWd0emk3Z3dmbDg2Y3RiY2UzYmZyM292OWl5eTZvd2gxdzhmOTM3eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uJ152PJ9aiwmwXF8Vv/giphy.webp", width=400)
    st.markdown("🚀 **Let's uncover the hidden patterns of online toxicity!**")

    # Add Dataset Information
    st.markdown("### About the Dataset 📊")
    st.markdown(
        """
        #### Key Points about the Cyberbully Detection Dataset

        1. **Dataset Purpose**:
            - Built to aid in developing and evaluating cyberbully detection models.
            - Covers various types of cyberbullying, including:
                - **Race/Ethnicity-related**: offensive content based on race or ethnicity.
                - **Gender/Sexual-related**: content targeting gender or sexual orientation.
                - **Religion-related**: offensive language based on religion.
                - **Non-cyberbullying**: Neutral or non-offensive content.
            - Enables researchers to create models to help foster safer online interactions.

        2. **Data Composition**:
            - **Total Records**: 99,990 tweets.
            - **Balanced Classes**:
                - Non-cyberbullying: 50,000 instances
                - Race/Ethnicity: 17,000 instances
                - Gender/Sexual: 17,000 instances
                - Religion: 16,000 instances
            - This balance across classes supports effective multi-class classification.

        3. **Data Format**:
            - **CSV Format** with columns:
                - **`text`**: Content of the tweet.
                - **`label`**: Type of cyberbullying or non-cyberbullying.

        4. **Applications**:
            - Useful for training machine learning and deep learning models in cyberbullying detection.
            - Balanced data aids in improving model accuracy and performance across different types of cyberbullying.

        5. **Community Contributions**:
            - Open for improvements and additions through community pull requests to continually enhance its utility.
        
        #### Dataset Summary
        - **Dataset Size**: 99,990 rows
        - **Features**:
            - **`text`**: Content of the social media post.
            - **`label`**: Category of cyberbullying (e.g., race/ethnicity, gender/sexual, religion, or non-cyberbullying).
        - **Purpose**: To aid in detecting and categorizing various forms of cyberbullying on social media.

        #### Usage
        Researchers and developers can use this dataset to explore machine learning and deep learning techniques aimed at automated cyberbullying detection, promoting a healthier online space.
        """
    )

    # Display Dataset Head
    st.markdown("### Sample of the Dataset 📝")
    st.dataframe(df.head())


elif pages == "📊 EDA":
    st.title("Exploratory Data Analysis (EDA) 📊")
    
    # What is EDA
    st.markdown(
        """
        ### What is EDA? 🤔
        Exploratory Data Analysis (EDA) is the process of examining datasets to summarize their main characteristics, often using visual methods. 
        It helps in:
        - Understanding the structure and distribution of the data.
        - Identifying patterns, anomalies, or relationships.
        - Preparing the data for further analysis or model building.
        
        For this dataset, EDA helps uncover trends in cyberbullying behaviors across different categories.
        """
    )

    # Class Distribution
    st.subheader("Class Distribution 📂")
    st.markdown(
        """
        **Why this visualization?**  
        Understanding the distribution of different types of cyberbullying helps ensure our dataset is balanced, 
        which is crucial for training machine learning models effectively.
        """
    )
    fig, ax = plt.subplots()
    sns.countplot(x='label', data=df, order=df['label'].value_counts().index, palette='coolwarm', ax=ax)
    ax.set_title('Class Distribution')
    ax.set_xlabel('Cyberbullying Category')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Tweet Length Distribution
    st.subheader("Tweet Length Distribution by Class ✍️")
    st.markdown(
        """
        **Why this visualization?**  
        Examining the length of tweets across different categories helps us understand content variability.
        Longer or shorter tweets might indicate differences in the way people engage in or respond to cyberbullying.
        """
    )
    df['tweet_length'] = df['text'].apply(len)
    fig, ax = plt.subplots()
    sns.boxplot(x='label', y='tweet_length', data=df, palette='coolwarm', ax=ax)
    ax.set_title('Tweet Length Distribution')
    ax.set_xlabel('Cyberbullying Category')
    ax.set_ylabel('Tweet Length (characters)')
    st.pyplot(fig)


elif pages == "💬 Word Analysis":
    st.title("Word Analysis 💬")
    
    # Introduction to Word Cloud
    st.markdown(
        """
        ### What is a Word Cloud? 🌟
        A **Word Cloud** is a visual representation of the most frequently occurring words in a given text. 
        In a word cloud, the size of each word represents its frequency — the more often a word appears, the larger it is displayed.
        
        **Why Word Clouds are Useful?**
        - **Pattern Recognition**: Word clouds help identify the most prominent words in a dataset. By visualizing frequent terms, 
        you can quickly understand the key topics or sentiments driving the data.
        - **Text Analysis**: They are particularly useful for text mining, where you need to extract meaning from a large amount of textual data.
        - **Sentiment Insights**: In the context of cyberbullying, word clouds can highlight commonly used abusive or offensive words, which 
        can be flagged and acted upon in moderation efforts.
        
        **For our dataset**: We will explore the most frequently used words within different cyberbullying categories such as 
        ethnicity, gender, and religion. This will give us an understanding of common language used for different types of bullying.
        """
    )
    
    # Select Cyberbullying Category
    category = st.selectbox("Choose a Cyberbullying Category", df['label'].unique())
    
    # Display the selected category
    st.markdown(f"### Word Cloud for **{category}**")
    st.markdown(
        f"Explore the most frequently used words in tweets categorized as **{category}**. Larger words represent higher frequency."
    )

    # Generate Word Cloud for the selected category
    text = ' '.join(df[df['label'] == category]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Set2').generate(text)
    
    # Display the Word Cloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')  # Hide axes for cleaner visualization
    st.pyplot(fig)
    
    # Additional Information on the Word Cloud
    st.markdown(
        """
        ### How to Interpret the Word Cloud? 🔍
        - **Larger Words**: Indicate that these words appear more frequently in tweets of the selected cyberbullying category.
        - **Smaller Words**: Represent words that appear less often but still contribute to the overall context.
        
        The word cloud helps to visually capture the most common terms associated with each type of cyberbullying. For example, 
        in **ethnicity-related cyberbullying**, you may notice terms related to derogatory labels or racial slurs, while **gender-related** 
        bullying may show specific offensive terms targeting gender identity.
        
        By examining the word cloud, we can detect which words are central to specific types of cyberbullying, making it easier to 
        identify harmful content in future posts.
        """
    )

elif pages == "📈 Advanced Visualizations":
    st.title("Advanced Visualizations 📈")
    
    # TF-IDF Analysis
    st.markdown(
        """
        ### TF-IDF (Term Frequency - Inverse Document Frequency) Analysis 📊
        
        **TF-IDF** is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. 
        It helps highlight the most relevant terms by considering both the frequency of the term in a document and its rarity across the entire dataset.
        
        **Why is TF-IDF useful for analyzing this dataset?**
        - **Identifying Key Terms**: TF-IDF can identify words that are particularly important within specific categories, 
        even if they appear less frequently across the entire dataset. This is useful for finding keywords related to specific 
        cyberbullying topics (e.g., ethnicity, religion, gender).
        - **Filtering Out Common Words**: Common words like 'the', 'is', or 'and' typically appear frequently across all documents, 
        but they do not carry much meaningful information. TF-IDF helps reduce the weight of these common words.
        - **Focus on Distinctive Words**: By calculating the TF-IDF score, you can find words that are characteristic of specific 
        cyberbullying types, allowing targeted analysis of offensive language or patterns in social media content.

        **In this analysis**, we will compute the top TF-IDF terms for each cyberbullying category. 
        This will help us understand which words are most relevant for detecting specific types of cyberbullying.
        """
    )
    
    # Perform TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
    
    # Create DataFrame for the TF-IDF matrix
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_df['label'] = df['label']
    
    # Compute the mean TF-IDF score for each term across categories
    tfidf_means = tfidf_df.groupby('label').mean().T
    
    # Display the TF-IDF mean scores as a dataframe
    st.subheader("Mean TF-IDF Scores by Category")
    st.dataframe(tfidf_means)
    
    # Generate and display the bar plot for TF-IDF scores
    st.subheader("Top TF-IDF Terms by Category")
    fig, ax = plt.subplots(figsize=(10, 6))  # Smaller figure for better fit
    tfidf_means.plot(kind='bar', ax=ax, color=sns.color_palette('coolwarm', n_colors=len(tfidf_means)))

    
    # Title and axis labels
    ax.set_title('Top TF-IDF Terms by Category', fontsize=16)
    ax.set_xlabel('Terms', fontsize=12)
    ax.set_ylabel('Mean TF-IDF Score', fontsize=12)
    
    # Rotate x-axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    st.pyplot(fig)

    st.markdown(
        """
        ### Interpreting the TF-IDF Bar Plot 🧐
        
        - **Y-Axis (Mean TF-IDF Score)**: The TF-IDF score represents the importance of each term. Higher scores indicate 
        that the term is more relevant or distinctive for that category. Lower scores mean the term is more common across the dataset 
        or less specific to the category.
        - **X-Axis (Terms)**: These are the most important terms (words) within each cyberbullying category. By examining the 
        categories like **ethnicity**, **religion**, and **gender**, you can see which words are most frequently associated with 
        specific forms of cyberbullying.

        The bar plot provides a clear visualization of the terms that contribute most to each cyberbullying category, helping 
        identify the language used in different contexts (e.g., racial slurs for **ethnicity**, derogatory terms for **gender**).

        **Why is this important?**
        - **Targeted Content Moderation**: By identifying key terms in cyberbullying, we can develop more effective algorithms for 
        automatic content moderation and flag offensive language.
        - **Improved Detection Models**: Understanding which words are central to different types of cyberbullying allows us to build 
        more accurate machine learning models for detecting and classifying harmful content.
        """
    )


elif pages == "🌌 Clustering":
    st.title("Clustering Visualizations 🌌")
    
    # Explanation of Clustering and Dimensionality Reduction
    st.markdown(
        """
        ## What is Clustering and Why is it Important for Cyberbullying Detection?
        
        Clustering is an unsupervised learning technique where we group similar data points together. It is useful for finding 
        patterns and structures in data that may not be immediately obvious. In the case of cyberbullying detection, clustering 
        helps us identify the underlying themes and types of harmful content based on similarities in the text data.

        **Dimensionality reduction** is an important technique for visualizing high-dimensional data. Cyberbullying text data 
        is often very high-dimensional, as each word in the dataset can be considered a feature. Dimensionality reduction methods 
        such as **PCA (Principal Component Analysis)** and **t-SNE (t-Distributed Stochastic Neighbor Embedding)** are used 
        to reduce the number of dimensions and project the data into two dimensions, making it easier to visualize and explore.
        """
    )
    
    # PCA (Principal Component Analysis)
    st.subheader("PCA Visualization (Principal Component Analysis)")
    
    st.markdown(
        """
        **Principal Component Analysis (PCA)** is a linear dimensionality reduction technique. It transforms the original features 
        into a new set of features called **principal components**. These components are ordered by the amount of variance they 
        explain in the data. The first principal component explains the most variance, and each subsequent component explains the 
        remaining variance in a decreasing order.

        **Why PCA?** In text data, there are typically thousands of features (words), making it hard to identify patterns. PCA 
        reduces this high-dimensional space into two components, allowing us to visualize the relationships between different 
        data points more easily. In our case, it helps in understanding how different types of cyberbullying are spread across 
        the text data.
        """
    )
    
    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=2)  # Reduce to 2 components for visualization
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])  # Transform the text data to TF-IDF matrix
    pca_result = pca.fit_transform(tfidf_matrix.toarray())  # Apply PCA to reduce dimensions

    # Plot PCA result
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['label'], palette='Set2', ax=ax)
    ax.set_title('PCA: Cyberbullying Categories', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.legend(title="Cyberbullying Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    st.pyplot(fig)
    
    st.markdown(
        """
        ### Interpreting the PCA Plot 🧐
        
        - **X and Y axes**: The two axes in the plot represent the two principal components, which explain the most variance 
        in the data.
        - **Colors**: Each color represents a different category of cyberbullying (e.g., ethnicity, gender, religion). Points 
        closer together represent similar text data, while points farther apart represent distinct types of content.
        
        **Why is PCA important here?** By reducing the dimensionality, we can easily see how the cyberbullying categories are 
        distributed and whether there are clear separations between them in the data. This visualization helps us understand the 
        diversity of language used for different types of cyberbullying.
        """
    )
    
    # t-SNE (t-Distributed Stochastic Neighbor Embedding)
    st.subheader("t-SNE Visualization (t-Distributed Stochastic Neighbor Embedding)")
    
    st.markdown(
        """
        **t-SNE (t-Distributed Stochastic Neighbor Embedding)** is a non-linear dimensionality reduction technique that focuses 
        on preserving the local structure of the data. Unlike PCA, which tries to explain variance in a linear way, t-SNE focuses 
        on keeping similar data points close to each other and dissimilar data points farther apart in the lower-dimensional space.

        **Why t-SNE?** t-SNE is particularly useful for visualizing complex data such as text because it can capture local 
        relationships that PCA might miss. It is great for visualizing clusters or patterns in high-dimensional data that might 
        not be linearly separable.
        """
    )
    
    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)  # Reduce to 2 components for visualization
    tsne_result = tsne.fit_transform(tfidf_matrix.toarray())  # Apply t-SNE to reduce dimensions
    
    # Plot t-SNE result
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=df['label'], palette='Set2', ax=ax)
    ax.set_title('t-SNE: Cyberbullying Categories', fontsize=16)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.legend(title="Cyberbullying Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    st.pyplot(fig)
    
    st.markdown(
        """
        ### Interpreting the t-SNE Plot 🧐
        
        - **X and Y axes**: The axes represent the two components in the t-SNE space, capturing the local structure of the data.
        - **Colors**: Each color corresponds to a specific category of cyberbullying, helping us differentiate between different 
        types of harmful content.
        
        **Why is t-SNE important here?** While PCA helps to explain global structures in the data, t-SNE is excellent for revealing 
        finer clusters or subcategories within the data. It can show how similar (or different) cyberbullying types are to one another 
        in terms of the language used, giving us a deeper understanding of the nuances in cyberbullying language.
        """
    )
