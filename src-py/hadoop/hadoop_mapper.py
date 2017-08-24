#!/usr/bin/python3
import sys
from random import randint

for line in sys.stdin:
    print('{}\t{}'.format(randint(0, int(10e8)), line.strip()))
