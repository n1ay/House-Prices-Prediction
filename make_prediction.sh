#!/bin/bash
PRICES_FILE='SalePrices.txt'
ID_FILE='Id.txt'
SUBMISSION_FILE='submission.csv'

echo "SalePrice" > $PRICES_FILE
python3.5 -O main.py | egrep -iv "(score)|(^$)" >> $PRICES_FILE
cat data/sample_submission.csv | awk -F"," '{ print $1 }' > $ID_FILE
paste $ID_FILE $PRICES_FILE -d "," > $SUBMISSION_FILE

rm $PRICES_FILE $ID_FILE

