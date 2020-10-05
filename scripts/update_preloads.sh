#!/bin/bash
echo "creating eeRepulsion.txt"
cat eeRepulsion.txt.* >> eeRepulsion.txt
rm eeRepulsion.txt.*
echo "creating coulomb.txt"
cat coulomb.txt.* >> coulomb.txt
rm coulomb.txt.*
