#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

resultDir='\\\\molphyschem\\DATA\\User\\DimuraM\\T4L\\PET_experiment_design\\QY_Thomas_all\\result\\'
def main():
  QPlist=range(120,165)
  LPlist=range(84,165)
  PDBlist=['t4l\\148l_leap_mod.pdb', 't4l\\F100-T300-II_RbAF100b027927_mod.pdb', 't4l\\F100-T330-II_c017148_mod.pdb', 't4l\\F50-T330_RbAhTF50b012479_mod.pdb', 't4l\\NMSIM_007859_mod.pdb', 't4l\\172l_leap_mod.pdb', 't4l\\F100-T300-I_RbAF100a043176_mod.pdb', 't4l\F100-T330-III_d011160_mod.pdb', 't4l\\forward-NMSim_6636_635_bestC3_mod.pdb', 't4l\\NMSIM_009447_mod.pdb', 't4l\\aMD-148l-all-047617_mod.pdb', 't4l\\F100-T330-I_a030399_mod.pdb', 't4l\\F50-T300_RbAF50c023769_mod.pdb', 't4l\\NMSIM_004694_mod.pdb']

  taskList=[]
  for lp in LPlist:
    for qp in QPlist:
      for pdb in PDBlist:
	taskList.append((pdb,lp,qp))

  pool = Pool(processes=6)
  result = pool.map(calcQY, taskList)
  pool.close()
  pool.join()
  
def putQuencher(resi,inFname,outFname):
  inf=open(inFname)
  outf=open(outFname, 'w')
  for line in inf:
    sz=line
    if line[:4]=="ATOM" and int(line[22:26])==resi:
      sz=line[:17]+"TRP"+line[20:]
    outf.write(sz)
    
def hideQuencher(resi,inFname,outFname):
  inf=open(inFname)
  outf=open(outFname, 'w')
  for line in inf:
    sz=line
    if line[:4]=="ATOM" and int(line[22:26])==resi:
      sz=line[:17]+"ALA"+line[20:]
    outf.write(sz)

def calcQY(tup):
  pdbPath=tup[0]
  lp=tup[1]
  qp=tup[2]
  
  pref=os.path.splitext(os.path.basename(pdbPath))[0]
  outpPdb = resultDir + 'QY_{}_lp{}_qp{}.txt'.format(pref,lp,qp)
  if os.path.isfile(outpPdb):
    print('{} already exists, skipping'.format(outpPdb))
    return
  
  tempPdb1 = tempfile.NamedTemporaryFile(suffix='.pdb', prefix='qy1_', delete=False)
  tempPdb2 = tempfile.NamedTemporaryFile(suffix='.pdb', prefix='qy2_', delete=False)
  tempPdb1.close()
  tempPdb2.close()
  
  putQuencher(qp,pdbPath,tempPdb1.name)
  hideQuencher(lp,tempPdb1.name,tempPdb2.name)
  
  
  subprocess.call(['python', 'determine_qy.py','-f',tempPdb2.name,'-c',' ','-p',str(lp),'-o',outpPdb], shell=True)
  os.remove(tempPdb1.name)
  os.remove(tempPdb2.name)

if __name__ == '__main__':
  main()
