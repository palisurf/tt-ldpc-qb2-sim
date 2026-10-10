#!/usr/bin/env python3
import sys

def main():
    lines = [l.strip().split() for l in open("matrices/AR4JA_r45_4c_128c_r12.chinn.out") if len(l.strip().split()) == 3]
    edges = [(int(l[0])-1, int(l[1])-1) for l in lines[:-1] if float(l[2]) > 0]
    
    v0_checks = [e[0] for e in edges if e[1] == 0]
    v2497_checks = [e[0] for e in edges if e[1] == 2497]
    print("v=0 checks:", v0_checks)
    print("v=2497 checks:", v2497_checks)

if __name__ == "__main__":
    main()
