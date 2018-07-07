# House-Prices-Prediction
Kaggle "House Prices: Advanced Regression Techniques" competition resolution.  
More info: [kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Competition Description
>![](https://kaggle2.blob.core.windows.net/competitions/kaggle/5407/media/housesbanner.png)
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

>With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

## My approach
  1. Encoding all non-numeric values to numeric using LabelEncoder
  2. Replacement of all NA/NaN using gaussian random
  3. Normalization (0-1 min-max scaling)
  4. Keras NN fit & predict
  
## Possibilities of performance enhancement:
  - Finding best fitting model with k-folding

## Acknowledgements
The [Ames Housing dataset](http://www.amstat.org/publications/jse/v19n3/decock.pdf) was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset.

Competition description is cited from [kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
