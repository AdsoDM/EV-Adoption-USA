# Clusterization Analysis of Electric Vehicle Adoption in the USA

# --- 1. SETUP: Load Libraries ---
# --------------------------------

# For PCA and clustering visualization
library(factoextra)
# For clustering algorithms like kmeans
library(cluster)
# For creating advanced plots
library(ggplot2)
# For data manipulation pipelines
library(dplyr)
# To prevent text labels from overlapping in plots
library(ggrepel)


# --- 2. DATA PREPARATION: All Years ---
# ---------------------------------------

# Load the complete and imputed dataset
# Note: The file path must be adjusted to your local machine.
ev_clean_df <- read.csv(file="E:/Oscar/DataScience/Kryterion/Trabajo final/EV Adoption USA/data/ev_adoption_clean.csv")

# For demonstration, let's assume 'ev_clean_df' is loaded.
# Encoding the 'Party' variable. Assigning 1 to Democratic party, assuming
# their policies are generally more aligned with environmental goals.
ev_clean_df <- ev_clean_df %>%
  mutate(Party = factor(Party, levels = c('Republican', 'Democratic'), labels = c(0, 1)))

# Convert the new 'Party' factor column first to character and then to numeric
ev_clean_df <- ev_clean_df %>%
  mutate(Party = as.numeric(as.character(Party)))


# --- 3. ANALYSIS ON 2023 DATA ---
# ---------------------------------
# The initial analysis on all years is complex to visualize.
# We will focus on the most recent year, 2023, for a clearer picture.

# Filter the dataset for the year 2023
ev_clean_2023_df <- ev_clean_df %>%
  filter(year == 2023)

# Set the 'state' column as row names for easier identification in plots
ev_clean_2023_df <- ev_clean_2023_df %>%
  column_to_rownames(var = "state")

# Select only numeric variables for PCA, excluding identifiers and constants
# 'fuel_economy' is constant for a given year, so it's removed.
numeric_data_2023 <- ev_clean_2023_df %>%
  select_if(is.numeric) %>%
  select(-Index, -year, -fuel_economy)


# --- 4. DIMENSIONALITY REDUCTION: PCA ---
# -----------------------------------------

# Perform Principal Component Analysis on the scaled data
pca_result_2023 <- prcomp(numeric_data_2023, scale = TRUE)
summary(pca_result_2023)

# Extract the scores for the first two principal components (PC1, PC2)
pca_scores_2023 <- as.data.frame(pca_result_2023$x[, 1:2])


# --- 5. CLUSTERING: K-MEANS ---
# -------------------------------

# Remove California, as it is a significant outlier (identified in initial analysis)
# that can distort the clustering results for other states.
#pca_scores_no_ca <- pca_scores_2023[rownames(pca_scores_2023) != "California", ]
#ev_clean_no_ca <- ev_clean_2023_df[rownames(ev_clean_2023_df) != "California", ]

# Scale the PCA scores before clustering
cluster_data <- scale(pca_scores_2023)

# Set a seed for reproducibility of the k-means algorithm
set.seed(123)

# Perform k-means clustering with k=2
kmeans_result <- kmeans(cluster_data, centers = 2, nstart = 25) # nstart improves stability

# --- 5.5. OUTLIER ANALYSIS: Find states most different from the mean ---
# -----------------------------------------------------------------------

# The 'cluster_data' is scaled, so its mean (centroid) is at (0,0).
# We calculate the Euclidean distance of each state from this center point.
# A larger distance indicates a greater deviation from the average profile.

# Calculate the distance for each state from the center (0,0)
distances_from_mean <- cluster_data %>%
  as.data.frame() %>%
  mutate(
    State = rownames(.),
    Distance = sqrt(PC1^2 + PC2^2)
  ) %>%
  # Arrange the states by distance in descending order
  arrange(desc(Distance))

# Display the top 4 states with the largest distance from the mean
cat("Top 4 states with the greatest difference from the dataset mean:\n")
print(head(distances_from_mean, 5))


# --- 6. VISUALIZATION: Clusters and Political Affiliation ---
# -------------------------------------------------------------

# Create a consolidated data frame for plotting
plot_data <- as.data.frame(cluster_data) %>%
  # Add a column for the cluster assignment from k-means
  mutate(Cluster = factor(kmeans_result$cluster)) %>%
  # Add a column for political affiliation with clear labels
  mutate(Party = factor(ev_clean_2023_df$Party,
                        levels = c(0, 1),
                        labels = c("Republican", "Democrat"))) %>%
  # Add state names for labeling
  mutate(State = rownames(cluster_data))

# Build the plot using ggplot2
ggplot(plot_data, aes(x = PC1, y = PC2, color = Party, shape = Cluster)) +
  # Layer 1: Draw the points
  geom_point(size = 4, alpha = 0.8) +
  
  # Layer 2: Add ellipses around the clusters
  stat_ellipse(aes(group = Cluster, color = NULL), linetype = "dashed", type = "norm") +
  
  # Layer 3: Add non-overlapping state labels
  geom_text_repel(aes(label = State),
                  color = "black",
                  size = 3,
                  show.legend = FALSE) +
  
  # --- Customization and Titles ---
  
  # Set custom colors to match political party conventions
  scale_color_manual(values = c("Republican" = "red", "Democrat" = "blue")) +
  
  # Set custom shapes for the clusters (16=circle, 17=triangle)
  scale_shape_manual(values = c("1" = 16, "2" = 17)) +
  
  # Add titles and labels
  labs(
    title = "Cluster Analysis of US States by EV Adoption Factors (2023)",
    subtitle = "Color = Political Party, Shape = Assigned Cluster",
    x = "Principal Component 1 (PC1)",
    y = "Principal Component 2 (PC2)",
    color = "Political Affiliation",
    shape = "K-Means Cluster"
  ) +
  
  # Use a clean theme and add reference lines
  theme_minimal() +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey")