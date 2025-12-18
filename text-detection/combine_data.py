import pandas as pd
import os

# Paths
imdb_path = "../dataset/imdb-renamed.csv"
tweets_path = "../dataset/tweets-renamed.csv"
output_path = "data.csv"

def combine_datasets():
    print("Loading datasets...")
    try:
        df_imdb = pd.read_csv(imdb_path)
        df_tweets = pd.read_csv(tweets_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"IMDB columns: {df_imdb.columns.tolist()}")
    print(f"Tweets columns: {df_tweets.columns.tolist()}")

    # Normalize columns
    # IMDB: id,label,tweet -> id, sentiment, text
    if 'label' in df_imdb.columns:
        df_imdb.rename(columns={'label': 'sentiment'}, inplace=True)
    if 'tweet' in df_imdb.columns:
        df_imdb.rename(columns={'tweet': 'text'}, inplace=True)

    # Tweets: id,sentiment,text -> already good
    
    # Select only necessary columns
    df_imdb = df_imdb[['sentiment', 'text']]
    df_tweets = df_tweets[['sentiment', 'text']]

    # Combine
    combined_df = pd.concat([df_imdb, df_tweets], ignore_index=True)
    
    # Clean
    combined_df['text'] = combined_df['text'].astype(str).fillna("")
    # Ensure sentiment is string or int, consistent
    combined_df['sentiment'] = combined_df['sentiment'].astype(str)

    print(f"Combined dataset size: {len(combined_df)}")
    
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined dataset to {output_path}")

if __name__ == "__main__":
    combine_datasets()
