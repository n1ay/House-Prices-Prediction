#!/bin/bash
PRICES_FILE='SalePrices.txt'
ID_FILE='Id.txt'
SUBMISSION_FILE='submission.csv'

regexp="(score)|(best estimator)|(=)"

echo "SalePrice" > $PRICES_FILE
result=$(python3.5 -O main.py)
echo "$result" | egrep -iv "(^$)|$regexp" >> $PRICES_FILE
cat data/sample_submission.csv | awk -F"," '{ print $1 }' > $ID_FILE
paste $ID_FILE $PRICES_FILE -d "," > $SUBMISSION_FILE

echo "$result" | egrep -i "$regexp"
rm $PRICES_FILE $ID_FILE

