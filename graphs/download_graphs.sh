#!/bin/bash

urls=(
    "Xhttps://suitesparse-collection-website.herokuapp.com/MM/INPRO/msdoor.tar.gz"
    "Xhttps://suitesparse-collection-website.herokuapp.com/MM/DNVS/shipsec1.tar.gz"
    "Xhttps://suitesparse-collection-website.herokuapp.com/MM/DNVS/shipsec5.tar.gz"
    "Xhttps://suitesparse-collection-website.herokuapp.com/MM/Boeing/ct20stif.tar.gz"
    "Xhttps://suitesparse-collection-website.herokuapp.com/MM/Nasa/nasasrb.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/hood.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Chen/pkustk11.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Chen/pkustk13.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/bmw3_2.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Bump_2911.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Mycielski/mycielskian17.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-LiveJournal.tar.gz"
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
