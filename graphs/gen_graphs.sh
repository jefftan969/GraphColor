#!/bin/sh

./gen_one -o dense1k -n 1000 -p 1 -s 0
./gen_one -o dense2k -n 2000 -p 1 -s 0
./gen_one -o sparse5k -n 5000 -p 0.02 -s 0
./gen_one -o sparse10k -n 10000 -p 0.01 -s 0
./gen_one -o sparse20k -n 20000 -p 0.005 -s 0
./gen_one -o sparse50k -n 50000 -p 0.002 -s 0
./gen_one -o sparse100k -n 100000 -p 0.001 -s 0
