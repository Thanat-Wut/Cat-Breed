# Cat-Breed Packaging Survey Analysis (AIE323)

This project contains a data preparation and analysis pipeline for a cat food packaging survey, developed for the AIE323 course. 
The goal is to determine the most preferred packaging design for a foreign cat food brand entering the Thai market.

## Project Structure
- `aie323_pipeline.py`: The main Python script that performs the full data pipeline.
- `BU Data from Survey Cases_final(5).csv`: The raw survey dataset.
- `cleaned_survey_data.csv`: The processed and cleaned dataset ready for further use.
- `chart*.png`: Visualizations of demographics, target distribution, feature correlations, and option mean scores.
- `presentation_summary.txt`: A text file containing the logic and cleanliness summary for presentation slides.

## Pipeline Steps
1. **Target Variable Identification**: Extracts the top 1 packaging choice for each respondent.
2. **Data Cleaning**: Drops missing values, handles logic checks, and standardizes text responses (Cat Breeds & Brands).
3. **Feature Selection & Encoding**: Encodes Likert scales and demographic data, then uses ANOVA to find the top 10 most significant features influencing the packaging choice.
4. **Data Visualization**: Generates charts for data exploration.
5. **Export & Summary**: Outputs the final cleaned dataset and summary text.

## Usage
Run the pipeline script using Python:
```bash
python aie323_pipeline.py
```
