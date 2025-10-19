import os
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()


def download_query_results(company_id: str, output_file: str, query_template: str):
    """
    Connects to a PostgreSQL database, runs a formatted query, and saves results as CSV.

    Parameters:
        company_id (str): The company identifier to inject into the SQL query
        output_file (str): Path to save the CSV file
        query_template (str): SQL query template string, must use {company_id}
    """
    conn = None
    try:
        query = query_template.format(company_id=company_id)
        # 1. Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        print("✅ Connected to the database successfully.")

        # 2. Run the query and load results into a pandas DataFrame
        df = pd.read_sql_query(query, conn)

        # 3. Save the DataFrame as CSV
        df.to_csv(output_file, index=False)
        print(f"📁 Query results saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        if conn:
            conn.close()


def download_subject_lines(company_id: str, output_file: str):
    query = """
        SELECT *
        FROM (
            SELECT id, type, experiment_id,
                   properties_values -> 'subject_line' ->> 'value' AS subject_line
            FROM variation
            WHERE company_id = '{company_id}'
        ) as t 
        WHERE subject_line IS NOT NULL;
    """
    download_query_results(company_id, output_file, query)


def download_links_to_creatives(company_id: str, output_file: str):
    query = """
        WITH links AS (
            SELECT id, type, experiment_id, 
                   properties_values -> 'html' -> 0 -> 'property' ->> 'value' AS cdn_link
            FROM variation
            WHERE company_id = '{company_id}'
        )
        SELECT *
        FROM links
        WHERE cdn_link LIKE 'https://cdn.eikona.io/%';
    """
    download_query_results(company_id, output_file, query)


def download_creative_images(links_file: str = "creative_pipeline.csv"):
    # Read the CSV file
    df = pd.read_csv(links_file)

    # Create the images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    for idx, row in df.iterrows():
        url = row["cdn_link"]
        # Extract file extension from URL
        parsed_url = urlparse(url)
        filename = f"{os.path.basename(parsed_url.path)}.jpg"
        # Save to images/ directory
        image_path = os.path.join("images", filename)
        if os.path.exists(image_path):
            print(f"Already exists, skipping: {filename}")
            continue
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
