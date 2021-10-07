#!/usr/bin/python
import sys

resi=int(sys.argv[1])
inFname=sys.argv[2]
outFname=sys.argv[3]

inf=open(inFname)
outf=open(outFname, 'w')
for line in inf:
  sz=line
  if line[:4]=="ATOM" and line[22:26]=='{:4d}'.format(resi):
    sz=line[:17]+"ALA"+line[20:]
  outf.write(sz)