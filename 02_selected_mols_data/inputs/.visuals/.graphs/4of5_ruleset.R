# =========================================================
# QED Distribution Analysis for Accepted vs Rejected Molecules
# =========================================================
#
# This script:
#   1. Loads QED values from two datasets
#   2. Bins the values into fixed QED intervals
#   3. Computes the percentage composition within each dataset
#   4. Generates a comparative bar plot
#
# The final plot shows how Accepted and Rejected molecules
# are distributed across the QED space.
#
# =========================================================


# =========================================================
# 1. Load Required Libraries
# =========================================================

library(readr)      # Fast CSV reading
library(dplyr)      # Data manipulation
library(ggplot2)    # Plotting
library(tidyr)      # Data tidying utilities
library(showtext)   # Custom font rendering


# =========================================================
# 2. Font Configuration
# =========================================================
#
# showtext allows ggplot2 to render external fonts correctly,
# including exported vector graphics and PDFs.
#
# "Public Sans" is imported from Google Fonts and registered
# locally under the alias "Aptos".
#
# =========================================================

showtext_auto()

font_add_google("Public Sans", "Aptos") 


# =========================================================
# 3. Load and Label Input Data
# =========================================================
#
# Only the QED column is imported from each CSV file.
#
# A new column called "Set" is added to preserve the origin
# of each molecule after merging both datasets.
#
# =========================================================

accepted <- read_csv(
  "accepted.csv",
  col_select = "QED"
) %>%
  mutate(Set = "Accepted")

rejected <- read_csv(
  "rejected.csv",
  col_select = "QED"
) %>%
  mutate(Set = "Rejected")


# =========================================================
# 4. Build Composition Table
# =========================================================
#
# Workflow:
#
#   a) Merge both datasets
#   b) Remove invalid QED values
#   c) Divide QED values into bins of width 0.05
#   d) Count molecules per bin and dataset
#   e) Convert counts into percentages
#
# Important:
# Percentages are calculated independently within each set
# (Accepted and Rejected sum separately to 100%).
#
# =========================================================

composition_df <- bind_rows(accepted, rejected) %>%
  
  # Keep only valid QED values inside the theoretical range
  filter(!is.na(QED), QED >= 0, QED <= 1) %>%
  
  # Create discrete QED intervals
  mutate(
    bin = cut(
      QED,
      breaks = seq(0, 1, by = 0.05),
      include.lowest = TRUE,
      right = FALSE
    )
  ) %>%
  
  # Remove possible undefined bins
  filter(!is.na(bin)) %>%
  
  # Count molecules per interval and dataset
  group_by(bin, Set) %>%
  summarise(n = n(), .groups = "drop_last") %>%
  
  # Convert counts to percentage composition
  mutate(percent = 100 * n / sum(n)) %>%
  
  ungroup()


# =========================================================
# 5. Generate Comparative Histogram
# =========================================================
#
# The plot compares the normalized QED composition of
# Accepted and Rejected molecules across all intervals.
#
# position_dodge() places bars side-by-side to facilitate
# direct comparison between both datasets.
#
# =========================================================

ggplot(composition_df, aes(x = bin, y = percent, fill = Set)) +
  
  # Bar plot using precomputed percentages
  geom_col(
    position = position_dodge(preserve = "single")
  ) +
  
  # =====================================================
# Manual Color and Legend Configuration
# =====================================================
#
# Colors can be modified using either:
#   - Hexadecimal codes
#   - Named R colors
#
# The legend title uses a mathematical expression to
# represent the selection criterion.
#
# =====================================================

scale_fill_manual(
  
  # Custom colors for each dataset
  values = c(
    "Accepted" = "#b8d8be",
    "Rejected" = "#ffaaa5"
  ),
  
  # Mathematical legend title
  name = expression(
    italic(V) ~ symbol("\307") ~
      group(
        "{",
        sum(italic(I)[i], i %in% "{L,G,E,M}"),
        "}"
      ) >= 3
  ),
  
  # Legend labels
  labels = c(
    Accepted = "Accepted molecules",
    Rejected = "Discarded molecules"
  )
) +
  
  # Axis labels
  labs(
    x = "QED Score",
    y = "Composition of the interval (%)"
  ) +
  
  # Base theme
  theme_classic() +
  
  # =====================================================
# Theme Customization
# =====================================================

theme(
  
  # Global font settings
  text = element_text(
    family = "Aptos",
    size = 12
  ),
  
  # Serif font improves mathematical notation readability
  legend.title = element_text(
    family = "serif",
    size = 15
  ),
  
  # Rotate x-axis labels for better spacing
  axis.text.x = element_text(
    angle = 45,
    hjust = 1
  )
)

# =========================================================
# End of Script
# =========================================================