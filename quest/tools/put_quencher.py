#!/usr/bin/python
import sys

resi=int(sys.argv[1])
inFname=sys.argv[2]
outFname=sys.argv[3]

inf=open(inFname)
outf=open(outFname, 'w')
for line in inf:
  sz=line
  if line[:4]=="ATOM" and int(line[22:26])==resi:
    sz=line[:17]+"TRP"+line[20:]
  outf.write(sz)
