#!/bin/bash
grep "^eeRepulsion.txt" *.log | sed "s/^eeRepulsion.txt //g" >> eeRepulsion.txt 
grep "^coulomb.txt" *.log | sed "s/^coulomb.txt //g" >> coulomb.txt 