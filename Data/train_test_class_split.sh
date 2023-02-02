#!/bin/bash

# Crea la cartella COVIDx-splitted
mkdir -p COVIDx-splitted

# Crea le cartelle per le classi all'interno di COVIDx-splitted/train
cd COVIDx-splitted
mkdir -p train
cd train
mkdir -p pneumonia COVID-19 normal
cd ../..

COUNTER=0

# Copia le immagini nella cartella giusta
while read line; do

  echo "$COUNTER"
  COUNTER=$((COUNTER+1))
  
  # Prendi le informazioni dalla riga corrente
  index=$(echo $line | awk '{print $1}')
  filename=$(echo $line | awk '{print $2}')
  class=$(echo $line | awk '{print $3}')

  # Copia l'immagine nella cartella giusta
  cp -n COVIDx/train/$filename COVIDx-splitted/train/$class/$filename
done < COVIDx/train_COVIDx9A.txt

# Sostituisci i nomi delle immagini nel nuovo file train_COVIDx9A.txt
echo "" > COVIDx-splitted/train_COVIDx9A.txt
while read line; do
  index=$(echo $line | awk '{print $1}')
  filename=$(echo $line | awk '{print $2}')
  class=$(echo $line | awk '{print $3}')
  other=$(echo $line | awk '{print $4}')

  echo "$index $class/$filename $class $other" >> COVIDx-splitted/train_COVIDx9A.txt
  
done < COVIDx/train_COVIDx9A.txt

# Ripeti le operazioni per la cartella test e il file test_COVIDx9A.txt
mkdir -p COVIDx-splitted/test
cd COVIDx-splitted/test
mkdir -p pneumonia COVID-19 normal
cd ../..

COUNTER=0

while read line; do
  index=$(echo $line | awk '{print $1}')
  filename=$(echo $line | awk '{print $2}')
  class=$(echo $line | awk '{print $3}')

  cp -n COVIDx/test/$filename COVIDx-splitted/test/$class/$filename

  echo "$COUNTER"
  COUNTER=$((COUNTER+1))

done < COVIDx/test_COVIDx9A.txt

echo "" > COVIDx-splitted/test_COVIDx9A.txt
while read line; do
  index=$(echo $line | awk '{print $1}')
  filename=$(echo $line | awk '{print $2}')
  class=$(echo $line | awk '{print $3}')
  other=$(echo $line | awk '{print $4}')

  echo "$index $class/$filename $class $other" >> COVIDx-splitted/test_COVIDx9A.txt
done < COVIDx/test_COVIDx9A.txt