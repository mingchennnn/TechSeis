import numpy as np
from scipy.special import rel_entr

def reorder_topics_by_weighted_mean_year(avg_vectors, reverse_order):
    """
    Reorders the columns of avg_vectors such that topics are sorted by the mean year,
    where the mean year is the weighted average of years weighted by the proportion for each topic.
    
    Parameters:
    avg_vectors (pd.DataFrame): DataFrame containing a 'year' column and topic columns with proportions.
    
    Returns:
    pd.DataFrame: Re-ordered DataFrame.
    """
    # Initialize a dictionary to store the mean years
    mean_years = {}
    topic_columns = [col for col in avg_vectors.columns if col.startswith('topic_')]
    # Loop over each topic column
    for topic in topic_columns:
        # Calculate the weighted mean year for the current topic
        weighted_sum = (avg_vectors['year'] * avg_vectors[topic]).sum()
        total_weight = avg_vectors[topic].sum()
        weighted_mean_year = weighted_sum / total_weight if total_weight != 0 else float('inf')
        mean_years[topic] = weighted_mean_year

    # Sort the topics by the weighted mean year
    sorted_topics = sorted(mean_years, key=mean_years.get, reverse=reverse_order)

    # Re-order the columns of the DataFrame
    new_column_order = ['year'] + sorted_topics
    avg_vectors_sorted = avg_vectors[new_column_order]

    return avg_vectors_sorted

def transform_avg_vector(df):
    """
    Transforms the df DataFrame so that the indices are topics 
    and the columns are years.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing a 'year' column and topic columns.
    
    Returns:
    pd.DataFrame: Transformed DataFrame with topics as indices and years as columns.
    """
    # Get the list of years
    years = df['year'].unique()
    
    # Set the 'year' column as the index
    df = df.set_index('year')
    
    # Transpose the DataFrame
    df_transposed = df.transpose()
    
    # Rename the columns to the unique years
    df_transposed.columns = years
    
    return df_transposed
    
def reorder_topics_by_weighted_mean_half_year(avg_vectors, reverse_order):
    """
    Reorders the columns of avg_vectors such that topics are sorted by the mean half-year,
    where the mean half-year is the weighted average of half-years weighted by the proportion for each topic.
    
    Parameters:
    avg_vectors (pd.DataFrame): DataFrame containing a 'year_half' column and topic columns with proportions.
    reverse_order (bool): Whether to sort topics in descending order of mean half-year.
    
    Returns:
    pd.DataFrame: Re-ordered DataFrame with topics sorted by weighted mean half-year.
    """
    # Initialize a dictionary to store the mean half-years
    mean_half_years = {}
    topic_columns = [col for col in avg_vectors.columns if col.startswith('topic_')]

    # Convert 'year_half' to numerical values for averaging (e.g., "1980-H1" becomes 1980.0, "1980-H2" becomes 1980.5)
    avg_vectors['year_half_numeric'] = avg_vectors['year_half'].apply(
        lambda x: float(x.split('-')[0]) + (0.5 if x.endswith('H2') else 0.0)
    )

    # Loop over each topic column
    for topic in topic_columns:
        # Calculate the weighted mean half-year for the current topic
        weighted_sum = (avg_vectors['year_half_numeric'] * avg_vectors[topic]).sum()
        total_weight = avg_vectors[topic].sum()
        weighted_mean_half_year = weighted_sum / total_weight if total_weight != 0 else float('inf')
        mean_half_years[topic] = weighted_mean_half_year

    # Sort the topics by the weighted mean half-year
    sorted_topics = sorted(mean_half_years, key=mean_half_years.get, reverse=reverse_order)

    # Re-order the columns of the DataFrame
    new_column_order = ['year_half'] + sorted_topics
    avg_vectors_sorted = avg_vectors[new_column_order]

    return avg_vectors_sorted

def transform_avg_vector_half_year(df):
    """
    Transforms the df DataFrame so that the indices are topics 
    and the columns are half-year periods.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing a 'year_half' column and topic columns.
    
    Returns:
    pd.DataFrame: Transformed DataFrame with topics as indices and half-year periods as columns.
    """
    # Get the list of half-year periods
    half_years = df['year_half'].unique()

    # Set the 'year_half' column as the index
    df = df.set_index('year_half')

    # Transpose the DataFrame
    df_transposed = df.transpose()

    # Rename the columns to the unique half-year periods
    df_transposed.columns = half_years

    return df_transposed


def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    if np.linalg.norm(v1) * np.linalg.norm(v2) != 0:
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        cos_theta = 0
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def compute_angle_differences(df):
    """
    Compute the angle differences between consecutive difference vectors.

    Parameters:
    df (pd.DataFrame): DataFrame with columns '1990', '1991', ..., '2023'.

    Returns:
    list: List of angle differences.
    """
    
    start_year = df.columns.tolist()[0]
    end_year = df.columns.tolist()[-1]
    angle_differences = []
    
    # Calculate the difference vectors
    difference_vectors = [df[year + 1] - df[year] for year in range(start_year, end_year)]

    # Calculate the angle differences between consecutive difference vectors
    for i in range(len(difference_vectors) - 1):
        angle_differences.append(calculate_angle(difference_vectors[i], difference_vectors[i+1]))

    return angle_differences


def compute_angle_differences_half_year(df):
    """
    Compute the angle differences between consecutive difference vectors for half-year periods.

    Parameters:
    df (pd.DataFrame): DataFrame with columns '1980-H1', '1980-H2', ..., '2024-H2'.

    Returns:
    list: List of angle differences.
    """
    
    # Extract sorted half-year periods to ensure correct chronological order
    half_years = sorted(df.columns.tolist())
    angle_differences = []
    # Calculate the difference vectors for consecutive half-year periods
    difference_vectors = [df[half_years[i + 1]] - df[half_years[i]] for i in range(len(half_years) - 1)]

    # Calculate the angle differences between consecutive difference vectors
    for i in range(len(difference_vectors) - 1):
        angle_differences.append(calculate_angle(difference_vectors[i], difference_vectors[i + 1]))

    return angle_differences



def gini_coefficient(x):
    """Calculate the Gini coefficient of a numpy array."""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    if len(x)**2 * np.mean(x) != 0:
        return diffsum / (len(x)**2 * np.mean(x))
    else:
        return 0


def insert_missing_years(vector_matrix, start_year, end_year):
    """Insert values for missing years"""
    # Check for missing years and fill them with zero values
    all_years = set(range(start_year, end_year + 1))
    present_years = set(vector_matrix.columns.astype(int))
    missing_years = all_years - present_years

    for year in missing_years:
        vector_matrix[year] = 0

    # Sort the columns to maintain the year order
    return vector_matrix.reindex(sorted(vector_matrix.columns.astype(int)), axis=1)

def insert_missing_half_years(vector_matrix, start_year, end_year):
    """Insert values for missing half-year periods."""
    # Generate all half-year periods from start_year to end_year
    all_half_years = [f"{year}-H{i}" for year in range(start_year, end_year + 1) for i in [1, 2]]
    present_half_years = set(vector_matrix.columns)
    missing_half_years = set(all_half_years) - present_half_years

    # Add missing half-year columns with zero values
    for half_year in missing_half_years:
        vector_matrix[half_year] = 0

    # Sort the columns to maintain the chronological half-year order
    return vector_matrix.reindex(sorted(all_half_years), axis=1)

def flexible_moving_average(rates, max_window):
    """Get a smoothed time series by averaging over a defined maximum window size"""
    length = len(rates)
    average_rates = np.zeros(length)
    # Calculate the average for each point considering the maximum available window
    for i in range(length):
        # Determine the start and end of the window around the current point
        window_start = max(0, i - max_window // 2)
        window_end = min(length, i + max_window // 2 + 1)
        # Calculate the average within the window
        average_rates[i] = np.mean(rates[window_start:window_end]) 
    return average_rates



# Kullback-Leibler Divergence
def kl_divergence(P, Q):
    return np.sum(rel_entr(P, Q))

# Jensen-Shannon Divergence
def js_divergence(P, Q):
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)






def reorder_clusters_by_weighted_mean_year(avg_vectors, reverse_order=False):
    """
    Reorders the rows of avg_vectors such that clusters are sorted by the weighted mean year,
    where the weighted mean year is the weighted average of years weighted by the proportion for each cluster.
    
    Parameters:
    avg_vectors (pd.DataFrame): DataFrame with 'cluster_id' column and year columns (e.g., '1930', '1931') containing proportions.
    reverse_order (bool): If True, sort in descending order of weighted mean year. Default is False (ascending).
    
    Returns:
    pd.DataFrame: Re-ordered DataFrame.
    """
    # Initialize a dictionary to store the mean years
    mean_years = {}
    
    # Get year columns (exclude cluster_id)
    year_columns = [col for col in avg_vectors.columns if col != 'cluster_id']
    # Convert year columns to numeric for calculations
    years = [int(year) for year in year_columns]
    
    # Loop over each cluster_id
    for _, row in avg_vectors.iterrows():
        cluster_id = row['cluster_id']
        # Calculate the weighted mean year for the current cluster
        weighted_sum = sum(row[year] * int(year) for year, year_val in zip(year_columns, years))
        total_weight = sum(row[year] for year in year_columns)
        weighted_mean_year = weighted_sum / total_weight if total_weight != 0 else float('inf')
        mean_years[cluster_id] = weighted_mean_year

    # Sort cluster_ids by the weighted mean year
    sorted_cluster_ids = sorted(mean_years, key=mean_years.get, reverse=reverse_order)

    # Re-order the rows of the DataFrame based on sorted cluster_ids
    avg_vectors_sorted = avg_vectors.set_index('cluster_id').loc[sorted_cluster_ids].reset_index()
    
    return avg_vectors_sorted



def compute_angle_differences_cluster(df):
    """
    Compute the angular differences between consecutive speed vectors of cluster distributions.
    
    Parameters:
    df (pd.DataFrame): DataFrame with 'cluster_id', 'earliness', and year columns (e.g., '1930', '1931', ...).
    
    Returns:
    list: Angular differences (in degrees) between consecutive speed vectors.
    """
    # Extract year columns, excluding 'cluster_id' and 'earliness'
    year_columns = [col for col in df.columns if col not in ['cluster_id', 'earliness']]
    
    # Get vectors for each year (proportions across all cluster_ids)
    vectors = [df[year].values for year in year_columns]
    
    # Compute speed vectors (difference between consecutive years)
    speed_vectors = [vectors[i+1] - vectors[i] for i in range(len(vectors)-1)]
    
    # Compute angular differences between consecutive speed vectors
    angular_differences = []
    for i in range(len(speed_vectors)-1):
        v1 = speed_vectors[i]
        v2 = speed_vectors[i+1]
        
        # Compute dot product and magnitudes
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            angle = 0.0  # If one vector is zero, assume no angular difference
        else:
            # Compute cosine of the angle
            cos_angle = dot_product / (norm_v1 * norm_v2)
            # Clip to avoid numerical errors outside [-1, 1]
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            # Compute angle in radians and convert to degrees
            angle = np.degrees(np.arccos(cos_angle))
        
        angular_differences.append(angle)
    
    return angular_differences
