
# Lab-4: Applied Statistical Modeling & Interactive Web Dashboard

**Dataset:** Medical Insurance Costs (`medical_insurance.csv`) — 2,772 raw rows, de-duplicated to 1,337 unique rows with 7 columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`.

## How to run

**Notebook (Parts 1 & 2):**
```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Dashboard (Part 3):**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Make sure `medical_insurance.csv` is in the same directory as `app.py`.

## Synthesis of findings

- **Smoking status** is by far the strongest driver of insurance charges — smokers' median charge (≈ $34,456) is roughly 4.7× non-smokers' (≈ $7,346), confirmed by a Mann-Whitney U test (p ≈ 5.7e-130) and an OLS coefficient of ≈ +$23,850 (p < 0.001).
- **Age** (+$257/year) and **BMI** (+$339/unit) both have smaller but statistically robust positive effects on charges.
- **Children** has a modest positive effect (+$475/child, p = 0.001).
- **Sex** has no significant effect on charges (p ≈ 0.70).
- **Region** shows, at best, a weak effect: a one-way ANOVA gives p ≈ 0.033 (technically significant), but Levene's test flags unequal variances across regions and the non-parametric Kruskal-Wallis alternative gives p ≈ 0.20 (not significant) — so region's effect should be treated cautiously.
- The OLS model explains **R² ≈ 0.751** (Adjusted R² ≈ 0.749) of the variance in charges. Diagnostics show mild heteroscedasticity and non-normal (right-skewed) residuals — both traceable to the right-skewed nature of `charges` itself, and a reasonable target for a follow-up log-transform.
- All Variance Inflation Factors for the continuous predictors are ≈ 1.0, so multicollinearity is not a concern.
