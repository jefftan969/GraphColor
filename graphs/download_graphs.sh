#!/bin/bash

urls=(
    "https://suitesparse-collection-website.herokuapp.com/MM/INPRO/msdoor.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DNVS/shipsec1.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DNVS/shipsec5.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Chen/pkustk11.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/ct20stif.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Chen/pkustk13.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Nasa/nasasrb.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/bmw3_2.tar.gz"
)

mkdir /tmp/jefftan-spethani-15418
cd /tmp/jefftan-spethani-15418
for url in ${urls[@]}; do
    wget $url;
    tar -xzf *.tar.gz;
    mv */*.mtx .;
    rm *.tar.gz;
    find . -type d -empty -delete
done
