#!/bin/bash
grep "^eeRepulsion.txt" *.log | sed "s/^.*log:eeRepulsion.txt //g" >> eeRepulsion.txt 
grep "^coulomb.txt" *.log | sed "s/^.*log:coulomb.txt //g" >> coulomb.txt 