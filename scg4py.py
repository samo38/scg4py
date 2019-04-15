#Module name:
#    scg4py (Systematic Coarse-Graining for Python)
#    (https://github.com/saeedMRT/scg4py)
#Author:
#    Saeed Mortezazadeh 

#For using this module you need python 3 and the following prerequisite modules: 
#numpy, scipy, matplotlib, and mdtraj for parsing the ’dcd’ and ’xtc’ trajectory file formats. 
#For ease of use, you can append the path of scg4py code to PYHTONPATH in the .bashrc file.
#
#export PYTHONPATH="/path/to/scg4py/:$PYTHONPATH"

import numpy as _np
from scipy import interpolate as _spint
from scipy import linalg as _linalg
from scipy.special import comb as _comb
from matplotlib import pyplot as _plt
import time as _time
import os as _os
import subprocess as _subP
import glob as _glob

#####################################################
#####################################################
def _isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False
#####################################################
#####################################################
def _Rstrip(inStr, Char):
    outStr = str(inStr)
    if isinstance(Char, list):
        pass
    else:
        Char = [Char]
    for i in range(len(Char)):
        if outStr.endswith(Char[i]):
            outStr = outStr[0: -1*len(Char[i])]
    return outStr
#####################################################
#####################################################
def _writeLMPtab(sysName, Btab, BONDtypeSet, Atab, ANGLEtypeSet, Dtab,
                 DIHEDRALtypeSet, NBtab, NonBONDEDtypeSet, refinePot=False):
    NB_tabName = sysName + '.tab_NB'
    B_tabName = sysName + '.tab_B'
    A_tabName = sysName + '.tab_A'
    D_tabName = sysName + '.tab_D'
    nPairType = len(NonBONDEDtypeSet)
    nBondType = len(BONDtypeSet)
    nAngleType = len(ANGLEtypeSet)
    nDihedralType = len(DIHEDRALtypeSet)
    if refinePot:
        space1 = '   '
        space2 = '      '
    else:
        space1 = ''
        space2 = '   '
    print(space1 + 'writing the tabulated potentials for LAMMPS simulation:')
    if nBondType > 0:
        with open(B_tabName,'w') as tabFile:
            print(space2 + B_tabName)
            tabFile.write('\n#################################################################\n')
            tabFile.write('######################## BOND POTENTIALS ########################\n')
            tabFile.write('#################################################################\n')
            for n in range(nBondType):
                tabFile.write('\nBond_{0:d}\n'.format(BONDtypeSet[n] + 1))
                x = Btab[n, 0][1:]
                pot = Btab[n, 1][1:]
                force = Btab[n, 2][1:]
                tabFile.write('N {0:d}\n\n'.format(len(x)))
                for i in _np.arange(len(x)):
                    tabFile.write('{0:5d} {1:12.6e} {2:.14e} {3:.14e}\n'.format(i + 1, x[i], pot[i], force[i]))
                tabFile.write('\n#################################################################\n')
    if nAngleType > 0:
        with open(A_tabName,'w') as tabFile:
            print(space2 + A_tabName)
            tabFile.write('\n#################################################################\n')
            tabFile.write('####################### ANGLE POTENTIALS ########################\n')
            tabFile.write('#################################################################\n')
            for n in range(nAngleType):
                tabFile.write('\nAngle_{0:d}\n'.format(ANGLEtypeSet[n] + 1))
                x = Atab[n, 0]
                pot = Atab[n, 1]
                force = Atab[n, 2]
                tabFile.write('N {0:d}\n\n'.format(len(x)))
                for i in _np.arange(len(x)):
                    tabFile.write('{0:5d} {1:12.6e} {2:.14e} {3:.14e}\n'.format(i + 1, x[i], pot[i], force[i]))
                tabFile.write('\n#################################################################\n')
    if nDihedralType > 0:
        with open(D_tabName,'w') as tabFile:
            print(space2 + D_tabName)
            tabFile.write('\n#################################################################\n')
            tabFile.write('###################### DIHEDRAL POTENTIALS ######################\n')
            tabFile.write('#################################################################\n')
            for n in range(nDihedralType):
                tabFile.write('\nDihedral_{0:d}\n'.format(DIHEDRALtypeSet[n] + 1))
                x = Dtab[n, 0][0:-1]
                pot = Dtab[n, 1][0:-1]
                force = Dtab[n, 2][0:-1]
                tabFile.write('N %d\n\n' % (_np.size(x)))
                for i in _np.arange(len(x)):
                    tabFile.write('{0:5d} {1:12.6e} {2:.14e} {3:.14e}\n'.format(i + 1, x[i], pot[i], force[i]))
                tabFile.write('\n#################################################################\n')
    with open(NB_tabName, 'w') as tabFile:
        print(space2 + NB_tabName)
        tabFile.write('\n#################################################################\n')
        tabFile.write('##################### NON-BONDED POTENTIALS #####################\n')
        tabFile.write('#################################################################\n')
        for n in range(nPairType):
            tabFile.write('\nNon-Bonded_{0:s}-{1:s}\n'.format(NonBONDEDtypeSet[n, 0], NonBONDEDtypeSet[n, 1]))
            x = NBtab[n, 0][1:]
            pot = NBtab[n, 1][1:]
            force = NBtab[n, 2][1:]
            tabFile.write('N {0:d} R {1:.6e} {2:.6e}\n\n'.format(len(x), x[0], x[-1]))
            for i in _np.arange(len(x)):
                tabFile.write('{0:5d} {1:12.6e} {2:.14e} {3:.14e}\n'.format(i + 1, x[i], pot[i], force[i]))
            tabFile.write('\n################################################\n')
        tabFile.write('\n#################################################################\n')
        tabFile.write('################ REPULSIVE NON-BONDED POTENTIALS ################\n')
        tabFile.write('#################################################################\n')
        tabFile.write('\n{0:s}\n'.format('NB_repulsive'))
        cutoff = x[-1]
        binNB = _np.mean(_np.diff(x))
        npoint = _np.int(_np.round(cutoff / binNB)) + 1
        x = _np.linspace(0, cutoff, npoint)[1:]
        maxNB = 1000
        rRep = 3.5
        a = float(maxNB) / (2 * rRep)
        pot = a * (x - rRep) ** 2
        force = -1 * 2 * a * (x - rRep)
        pot[x > rRep] = 0.0
        force[x > rRep] = 0.0
        tabFile.write('N {0:d} R {1:.6e} {2:.6e}\n\n'.format(len(x), x[0], x[-1]))
        for i in _np.arange(len(x)):
            tabFile.write('{0:5d} {1:12.6e} {2:.14e} {3:.14e}\n'.format(i + 1, x[i], pot[i], force[i]))
        tabFile.write('\n################################################\n')
#####################################################
#####################################################
class _scg4pyError(Exception):
    pass
#####################################################
#####################################################
class _cMAP:
    def __init__(self, inFile):
        fid = open(inFile, 'r')
        mapFile = fid.readlines()
        fid.close()
        self.AATRAJ = None
        self.CGTRAJ = None
        trj_types = ['.gro', '.pdb', '.lammpstrj', '.xtc', '.dcd']
        for line in mapFile:
            if 'AATRAJ' in line.split('#')[0]:
                lsplit = line.split('#')[0].split('=')[1].strip()
                if any([lsplit.endswith(i) for i in trj_types]):
                    self.AATRAJ = lsplit
                else:
                    mes = '\nAtomistic trajectory should be one of :\n [".gro", ".pdb", ".lammpstrj", ".xtc", ".dcd"]'
                    raise _scg4pyError(mes)
            elif 'CGTRAJ' in line.split('#')[0]:
                lsplit = line.split('#')[0].split('=')[1].strip()
                if any([lsplit.endswith(i) for i in trj_types]):
                    self.CGTRAJ = lsplit
                else:
                    mes = '\nCoarse-Grained trajectory should be one of :\n [".gro", ".pdb", ".lammpstrj", ".xtc"]'
                    raise _scg4pyError(mes)
            elif 'MOL' in line.split('#')[0].upper():
                lsplit0 = line.split('#')[0].split('=')[0].strip()
                lsplit1 = line.split('#')[0].split('=')[1].strip()
                if lsplit0.upper() == 'MOL':
                    self.MOL = _np.array(lsplit1.split(), dtype='<U5')
                elif lsplit0.upper() == 'NMOL':
                    self.nMOL = _np.array(lsplit1.split(), dtype=int)
        if len(self.MOL) > len(_np.unique(self.MOL)):
            mes = 'redundancy in the molecules name.'
            raise _scg4pyError(mes)
        if len(self.MOL) != len(self.nMOL):
            raise ValueError('The length of "MOL" and "NMOL" entry should be equal.\n')
        self.ATOMch = _np.zeros(len(self.MOL), dtype=object)
        self.ATOMms = _np.zeros(len(self.MOL), dtype=object)
        self.nATOM = _np.zeros(len(self.MOL), dtype=int)
        self.BEADname = _np.zeros(len(self.MOL), dtype=object)
        self.BEADtype = _np.zeros(len(self.MOL), dtype=object)
        self.LMPtype = _np.zeros(len(self.MOL), dtype=object)
        self.BEADmap = _np.zeros(len(self.MOL), dtype=object)
        self.BEADch = _np.zeros(len(self.MOL), dtype=object)
        self.BEADms = _np.zeros(len(self.MOL), dtype=object)
        self.nBEAD = _np.zeros(len(self.MOL), dtype=int)
        self.BONDterm = _np.zeros(len(self.MOL), dtype=bool)
        self.BONDtype = _np.zeros(len(self.MOL), dtype=object)
        self.nBONDtype = _np.zeros(len(self.MOL), dtype=int)
        self.BONDtypeName = _np.zeros(len(self.MOL), dtype=object)
        self.BONDtypeIdx = _np.zeros(len(self.MOL), dtype=object)
        self.ANGLEterm = _np.zeros(len(self.MOL), dtype=bool)
        self.ANGLEtype = _np.zeros(len(self.MOL), dtype=object)
        self.nANGLEtype = _np.zeros(len(self.MOL), dtype=int)
        self.ANGLEtypeName = _np.zeros(len(self.MOL), dtype=object)
        self.ANGLEtypeIdx = _np.zeros(len(self.MOL), dtype=object)
        self.DIHEDRALterm = _np.zeros(len(self.MOL), dtype=bool)
        self.DIHEDRALtype = _np.zeros(len(self.MOL), dtype=object)
        self.nDIHEDRALtype = _np.zeros(len(self.MOL), dtype=int)
        self.DIHEDRALtypeName = _np.zeros(len(self.MOL), dtype=object)
        self.DIHEDRALtypeIdx = _np.zeros(len(self.MOL), dtype=object)
        for mol in range(len(self.MOL)):
            nbond_mol = 0
            line = 0
            while line < len(mapFile):
                lsplit0 = mapFile[line].split('#')[0].split('=')[0].strip().split(':')
                if len(lsplit0) == 3:
                    lsplit1 = mapFile[line].split('#')[0].split('=')[1].strip().upper()
                    lsplit0[0] = lsplit0[0].upper()
                    lsplit0[2] = lsplit0[2].upper()
                    if lsplit0[0] == 'MOL' and (lsplit0[2] == 'ATOM' or lsplit0[2] == 'MAPPING' or
                                                lsplit0[2] == 'BOND' or lsplit0[2] == 'ANGLE' or lsplit0[
                                                    2] == 'DIHEDRAL'):
                        if lsplit0[1] in self.MOL:
                            pass
                        else:
                            mes = 'error in line {}.'.format(line + 1)
                            raise _scg4pyError(mes)
                    if lsplit0[0] == 'MOL' and lsplit0[1] == self.MOL[mol] and lsplit0[2] == 'ATOM':
                        natom = int(lsplit1)
                        n = 0
                        line += 1
                        atom_ch = _np.zeros(natom, dtype=float)
                        atom_ms = _np.zeros(natom, dtype=float)
                        while n < natom:
                            lco = mapFile[line].split('#')[0]
                            if 'MOL' in lco.upper():
                                raise TypeError('Error in "MOL:{}:ATOM" section'.format(self.MOL[mol]))
                            elif len(lco.split()) == 4:
                                try:
                                    atom_ch[n] = lco.split()[2]
                                    atom_ms[n] = lco.split()[3]
                                    n += 1
                                except:
                                    pass
                                line += 1
                            else:
                                line += 1
                        self.ATOMch[mol] = atom_ch
                        self.ATOMms[mol] = atom_ms
                        self.nATOM[mol] = natom
                    elif lsplit0[0] == 'MOL' and lsplit0[1] == self.MOL[mol] and lsplit0[2] == 'MAPPING':
                        nbead = int(lsplit1)
                        n = 0
                        line += 1
                        bead_name = _np.zeros(nbead, dtype='<U5')
                        bead_type = _np.zeros(nbead, dtype='<U5')
                        bead_map = _np.zeros(nbead, dtype=object)
                        while n < nbead:
                            lco = mapFile[line].split('#')[0]
                            if 'MOL' in lco:
                                raise TypeError('Error in "MOL:{}:MAPPING" section'.format(self.MOL[mol]))
                            elif len(lco.split(':')[0].split()) == 3:
                                try:
                                    bead_name[n] = lco.split(':')[0].split()[1]
                                    bead_type[n] = lco.split(':')[0].split()[2]
                                    bead_map[n] = _np.array(lco.split(':')[1].split(), dtype=int)
                                    n += 1
                                except:
                                    pass
                                line += 1
                            else:
                                line += 1
                        self.BEADname[mol] = bead_name
                        self.BEADtype[mol] = bead_type
                        self.BEADmap[mol] = bead_map - 1
                        self.nBEAD[mol] = nbead
                    elif lsplit0[0] == 'MOL' and lsplit0[1] == self.MOL[mol] and lsplit0[2] == 'BOND':
                        nbond_type = int(lsplit1)
                        if nbond_type > 0:
                            self.BONDterm[mol] = True
                        elif nbond_type == 0:
                            self.BONDterm[mol] = False
                        else:
                            raise TypeError('Error in "MOL:{}:BOND" section'.format(self.MOL[mol]))
                        n = 0
                        line += 1
                        bond_type = _np.zeros(nbond_type, dtype=int)
                        bond_type_idx = _np.zeros(nbond_type, dtype=object)
                        while n < nbond_type:
                            lco = mapFile[line].split('#')[0]
                            if 'MOL' in lco:
                                raise TypeError('Error in "MOL:{}:BOND" section'.format(self.MOL[mol]))
                            elif len(lco.split(':')[0].split()) == 1:
                                try:
                                    bond_type[n] = int(lco.split(':')[0])
                                    temp = lco.split(':')[1].split(',')
                                    temp = [i.split() for i in temp if bool(i.split())]
                                    bond_type_idx[n] = _np.array(temp, dtype=int)
                                    n += 1
                                except:
                                    pass
                                line += 1
                            else:
                                line += 1
                        if nbond_type > 0:
                            self.BONDtype[mol] = bond_type - 1
                            self.BONDtypeIdx[mol] = bond_type_idx - 1
                        else:
                            self.BONDtype[mol] = _np.array([])
                            self.BONDtypeIdx[mol] = _np.array([])
                        self.nBONDtype[mol] = nbond_type
                        for i in range(nbond_type):
                            nbond_mol = nbond_mol + len(self.BONDtypeIdx[mol][i])
                    elif lsplit0[0] == 'MOL' and lsplit0[1] == self.MOL[mol] and lsplit0[2] == 'ANGLE':
                        if lsplit1 == 'YES':
                            if nbond_mol > 1:
                                self.ANGLEterm[mol] = True
                            else:
                                self.ANGLEterm[mol] = False
                            if not self.BONDterm[mol]:
                                self.ANGLEterm[mol] = False
                        elif lsplit1 == 'NO':
                            self.ANGLEterm[mol] = False
                        else:
                            raise TypeError('Error in "MOL:{}:ANGLE" section'.format(self.MOL[mol]))
                    elif lsplit0[0] == 'MOL' and lsplit0[1] == self.MOL[mol] and lsplit0[2] == 'DIHEDRAL':
                        if lsplit1 == 'YES':
                            if nbond_mol > 2:
                                self.DIHEDRALterm[mol] = True
                            else:
                                self.DIHEDRALterm[mol] = False
                            if not self.ANGLEterm[mol] or not self.BONDterm[mol]:
                                self.DIHEDRALterm[mol] = False
                        elif lsplit1 == 'NO':
                            self.DIHEDRALterm[mol] = False
                        else:
                            raise TypeError('Error in "MOL:{}:DIHEDRAL" section'.format(self.MOL[mol]))
                line += 1

    def do_classification(self):
        TotAngles = _np.zeros(len(self.MOL), dtype=object)
        TotAnglesType = _np.zeros(len(self.MOL), dtype=object)
        TotAnglesTypeClass = _np.zeros([0, 2], dtype=int)

        TotDihedral = _np.zeros(len(self.MOL), dtype=object)
        TotDihedralType = _np.zeros(len(self.MOL), dtype=object)
        TotDihedralTypeClass = _np.zeros([0, 3], dtype=int)

        for mol in range(len(self.MOL)):
            beadms = _np.zeros(self.nBEAD[mol])
            beadch = _np.zeros(self.nBEAD[mol])
            for n in range(self.nBEAD[mol]):
                ind = self.BEADmap[mol][n]
                mass = _np.sum(self.ATOMms[mol][ind])
                charge = _np.sum(self.ATOMch[mol][ind])
                if _np.round(charge, 3) == 0.0:
                    charge = 0.0
                beadms[n] = _np.round(mass, 4)
                beadch[n] = _np.round(charge, 3)
            self.BEADch[mol] = beadch
            self.BEADms[mol] = beadms
            ############################################
            ############################################
            bondtype_name = _np.zeros([self.nBONDtype[mol], 2], dtype='U5')
            for i in range(self.nBONDtype[mol]):
                ind = self.BONDtypeIdx[mol][i][0]
                A = self.BEADtype[mol][ind[0]]
                B = self.BEADtype[mol][ind[1]]
                bondtype_name[i] = [A, B]
            self.BONDtypeName[mol] = bondtype_name

            nAllBond = 0
            for i in range(self.nBONDtype[mol]):
                nAllBond = nAllBond + len(self.BONDtypeIdx[mol][i])
            allBOND = _np.zeros([nAllBond, 2], dtype=int)
            n = 0
            for i in range(self.nBONDtype[mol]):
                m = len(self.BONDtypeIdx[mol][i])
                allBOND[n:n + m] = self.BONDtypeIdx[mol][i]
                n += m

            BONDtree = _np.zeros(self.nBEAD[mol], dtype=object)
            for bead in range(self.nBEAD[mol]):
                maplist = _np.array([], dtype=int)
                maplist = _np.append(maplist, allBOND[allBOND[:, 0] == bead][:, 1])
                maplist = _np.append(maplist, allBOND[allBOND[:, 1] == bead][:, 0])
                maplist = _np.unique(maplist)
                BONDtree[bead] = maplist
            ############################################
            ############################################
            if self.ANGLEterm[mol]:
                m = int(_comb(nAllBond, 2))
                allangle = _np.zeros([m, 3], dtype=int)
                n = 0
                for i in range(self.nBEAD[mol]):
                    for j in BONDtree[i]:
                        if j != i:
                            for k in BONDtree[j]:
                                if k != i and k != j:
                                    cmp1 = _np.any(_np.all(allBOND == [i, k], axis=1))
                                    cmp2 = _np.any(_np.all(allBOND == [k, i], axis=1))
                                    if not (cmp1 or cmp2):
                                        cmp1 = _np.any(_np.all(allangle[0:n] == [i, j, k], axis=1))
                                        cmp2 = _np.any(_np.all(allangle[0:n] == [k, j, i], axis=1))
                                        if not (cmp1 or cmp2):
                                            allangle[n, :] = [i, j, k]
                                            n += 1
                TotAngles[mol] = _np.delete(allangle, range(n, m), axis=0)

                allangleType = _np.zeros([len(TotAngles[mol]), 2], dtype=int)
                for i in range(len(TotAngles[mol])):
                    angId_1 = TotAngles[mol][i, 0]
                    angId_2 = TotAngles[mol][i, 1]
                    angId_3 = TotAngles[mol][i, 2]
                    for bdT in range(len(self.BONDtypeIdx[mol])):
                        comp1 = [angId_1, angId_2] == self.BONDtypeIdx[mol][bdT]
                        comp2 = [angId_2, angId_1] == self.BONDtypeIdx[mol][bdT]
                        cmp1 = _np.any(_np.all(comp1, axis=1))
                        cmp2 = _np.any(_np.all(comp2, axis=1))
                        if cmp1 or cmp2:
                            allangleType[i, 0] = self.BONDtype[mol][bdT]
                        comp1 = [angId_2, angId_3] == self.BONDtypeIdx[mol][bdT]
                        comp2 = [angId_3, angId_2] == self.BONDtypeIdx[mol][bdT]
                        cmp1 = _np.any(_np.all(comp1, axis=1))
                        cmp2 = _np.any(_np.all(comp2, axis=1))
                        if cmp1 or cmp2:
                            allangleType[i, 1] = self.BONDtype[mol][bdT]
                TotAnglesType[mol] = allangleType

                m = len(TotAnglesType[mol])
                allangleTypeClass = _np.zeros([m, 2], dtype=int)
                n = 0
                for i in range(m):
                    ind1 = TotAnglesType[mol][i, 0]
                    ind2 = TotAnglesType[mol][i, 1]
                    cmp1 = _np.any(_np.all(allangleTypeClass[0:n] == [ind1, ind2], axis=1))
                    cmp2 = _np.any(_np.all(allangleTypeClass[0:n] == [ind2, ind1], axis=1))
                    if not (cmp1 or cmp2):
                        allangleTypeClass[n] = [ind1, ind2]
                        n += 1
                allangleTypeClass = _np.delete(allangleTypeClass, range(n, m), axis=0)
                self.nANGLEtype[mol] = len(allangleTypeClass)
                TotAnglesTypeClass = _np.append(TotAnglesTypeClass, allangleTypeClass, axis=0)
            else:
                self.nANGLEtype[mol] = 0
            ############################################
            ############################################
            if self.DIHEDRALterm[mol]:
                m = int(_comb(nAllBond, 3))
                alldihedral = _np.zeros([m, 4], dtype=int)
                n = 0
                for i in range(self.nBEAD[mol]):
                    for j in BONDtree[i]:
                        if j != i:
                            for k in BONDtree[j]:
                                if k != j and k != i:
                                    for l in BONDtree[k]:
                                        if l != i and l != j and l != k:
                                            cmp1 = _np.any(_np.all(allBOND == [i, l], axis=1))
                                            cmp2 = _np.any(_np.all(allBOND == [l, i], axis=1))
                                            if not (cmp1 or cmp2):
                                                comp1 = alldihedral[0:n] == [i, j, k, l]
                                                comp2 = alldihedral[0:n] == [l, k, j, i]
                                                cmp1 = _np.any(_np.all(comp1, axis=1))
                                                cmp2 = _np.any(_np.all(comp2, axis=1))
                                                if not (cmp1 or cmp2):
                                                    alldihedral[n, :] = [i, j, k, l]
                                                    n += 1
                TotDihedral[mol] = _np.delete(alldihedral, range(n, m), axis=0)

                alldihedralType = _np.zeros([len(TotDihedral[mol]), 3], dtype=int)
                for i in range(_np.size(TotDihedral[mol], axis=0)):
                    dihId_1 = TotDihedral[mol][i, 0]
                    dihId_2 = TotDihedral[mol][i, 1]
                    dihId_3 = TotDihedral[mol][i, 2]
                    dihId_4 = TotDihedral[mol][i, 3]
                    for bdT in range(len(self.BONDtypeIdx[mol])):
                        comp1 = [dihId_1, dihId_2] == self.BONDtypeIdx[mol][bdT]
                        comp2 = [dihId_2, dihId_1] == self.BONDtypeIdx[mol][bdT]
                        cmp1 = _np.any(_np.all(comp1, axis=1))
                        cmp2 = _np.any(_np.all(comp2, axis=1))
                        if cmp1 or cmp2:
                            alldihedralType[i, 0] = self.BONDtype[mol][bdT]
                        comp1 = [dihId_2, dihId_3] == self.BONDtypeIdx[mol][bdT]
                        comp2 = [dihId_3, dihId_2] == self.BONDtypeIdx[mol][bdT]
                        cmp1 = _np.any(_np.all(comp1, axis=1))
                        cmp2 = _np.any(_np.all(comp2, axis=1))
                        if cmp1 or cmp2:
                            alldihedralType[i, 1] = self.BONDtype[mol][bdT]
                        comp1 = [dihId_3, dihId_4] == self.BONDtypeIdx[mol][bdT]
                        comp2 = [dihId_4, dihId_3] == self.BONDtypeIdx[mol][bdT]
                        cmp1 = _np.any(_np.all(comp1, axis=1))
                        cmp2 = _np.any(_np.all(comp2, axis=1))
                        if cmp1 or cmp2:
                            alldihedralType[i, 2] = self.BONDtype[mol][bdT]
                TotDihedralType[mol] = alldihedralType

                m = len(alldihedralType)
                dihedralTypeClass = _np.zeros([m, 3], dtype=int)
                n = 0
                for i in range(m):
                    ind1 = TotDihedralType[mol][i, 0]
                    ind2 = TotDihedralType[mol][i, 1]
                    ind3 = TotDihedralType[mol][i, 2]
                    cmp1 = _np.any(_np.all(dihedralTypeClass[0:n] == [ind1, ind2, ind3], axis=1))
                    cmp2 = _np.any(_np.all(dihedralTypeClass[0:n] == [ind3, ind2, ind1], axis=1))
                    if not (cmp1 or cmp2):
                        dihedralTypeClass[n] = [ind1, ind2, ind3]
                        n += 1
                dihedralTypeClass = _np.delete(dihedralTypeClass, range(n, m), axis=0)
                self.nDIHEDRALtype[mol] = len(dihedralTypeClass)
                TotDihedralTypeClass = _np.append(TotDihedralTypeClass, dihedralTypeClass, axis=0)
            else:
                self.nDIHEDRALtype[mol] = 0
        #################################################################
        #################################################################
        massSeq = _np.concatenate(self.BEADms[:])
        typeSeq = _np.concatenate(self.BEADtype[:])
        lmpIdSeq = _np.zeros(len(typeSeq), dtype=int)
        typeGroup = []
        for i in range(len(typeSeq)):
            if typeSeq[i] not in typeGroup:
                typeGroup = _np.append(typeGroup, typeSeq[i])
        tn = 1
        for i in range(len(typeGroup)):
            ind1 = typeSeq == typeGroup[i]
            m_ind1_grp = []
            for J in massSeq[ind1]:
                if J not in m_ind1_grp:
                    m_ind1_grp = _np.append(m_ind1_grp, J)
            for k in range(len(m_ind1_grp)):
                ind2 = massSeq == m_ind1_grp[k]
                ind = _np.logical_and(ind1, ind2)
                lmpIdSeq[ind] = tn
                tn += 1
        ind = _np.append(0, _np.cumsum(self.nBEAD))
        for mol in range(len(self.MOL)):
            self.LMPtype[mol] = lmpIdSeq[ind[mol]:ind[mol + 1]]
        #################################################################
        #################################################################
        m = len(TotAnglesTypeClass)
        temp = _np.zeros([m, 2], dtype=int)
        n = 0
        for i in range(m):
            ind1 = TotAnglesTypeClass[i, 0]
            ind2 = TotAnglesTypeClass[i, 1]
            cmp1 = _np.any(_np.all(temp[0:n] == [ind1, ind2], axis=1))
            cmp2 = _np.any(_np.all(temp[0:n] == [ind2, ind1], axis=1))
            if not (cmp1 or cmp2):
                temp[n] = [ind1, ind2]
                n += 1
        TotAnglesTypeClass = _np.delete(temp, range(n, m), axis=0)

        for mol in range(len(self.MOL)):
            IDs_angletype = _np.zeros(self.nANGLEtype[mol], dtype=object)
            angletype = _np.zeros(self.nANGLEtype[mol], dtype=int)
            angletype_name = _np.zeros([self.nANGLEtype[mol], 3], dtype='U5')
            if self.nANGLEtype[mol] > 0:
                n = 0
                for i in range(len(TotAnglesTypeClass)):
                    IDsTemp = _np.zeros([0, 3], dtype=int)
                    indT1 = TotAnglesTypeClass[i, 0]
                    indT2 = TotAnglesTypeClass[i, 1]
                    redundant = _np.all([indT1, indT2] == [indT2, indT1])
                    comp1 = TotAnglesType[mol] == [indT1, indT2]
                    comp2 = TotAnglesType[mol] == [indT2, indT1]
                    cmp1 = _np.all(comp1, axis=1)
                    cmp2 = _np.all(comp2, axis=1)
                    if _np.any(cmp1):
                        temp = TotAngles[mol][cmp1]
                        IDsTemp = _np.concatenate([IDsTemp, temp], axis=0)
                        ind1 = temp[0, 0]
                        ind2 = temp[0, 1]
                        ind3 = temp[0, 2]
                        A = self.BEADtype[mol][ind1]
                        B = self.BEADtype[mol][ind2]
                        C = self.BEADtype[mol][ind3]
                        angletype[n] = i
                        angletype_name[n] = [A, B, C]
                        IDs_angletype[n] = IDsTemp
                        n += 1
                    if _np.any(cmp2) and not redundant:
                        temp = TotAngles[mol][cmp2]
                        IDsTemp = _np.concatenate([IDsTemp, temp], axis=0)
                        ind1 = temp[0, 0]
                        ind2 = temp[0, 1]
                        ind3 = temp[0, 2]
                        A = self.BEADtype[mol][ind1]
                        B = self.BEADtype[mol][ind2]
                        C = self.BEADtype[mol][ind3]
                        angletype[n] = i
                        angletype_name[n] = [A, B, C]
                        IDs_angletype[n] = IDsTemp
                        n += 1
                    # IDsTemp = _np.delete(IDsTemp, 0, axis=0)
                    # IDs_angletype[n] = IDsTemp
                    # if _np.any(cmp1) or _np.any(cmp2):
                    #     n += 1
                self.ANGLEtypeIdx[mol] = IDs_angletype
                self.ANGLEtype[mol] = angletype
                self.ANGLEtypeName[mol] = angletype_name
            else:
                pass
                # print('There is no angle interaction in the "' + self.MOL[mol] + '" molecule.')
        ########################################
        ########################################
        m = len(TotDihedralTypeClass)
        temp = _np.zeros([m, 3], dtype=int)
        n = 0
        for i in range(m):
            ind = TotDihedralTypeClass[i]
            cmp1 = _np.any(_np.all(temp[0:n] == ind, axis=1))
            cmp2 = _np.any(_np.all(temp[0:n] == [ind[2], ind[1], ind[0]], axis=1))
            if not (cmp1 or cmp2):
                temp[n] = ind
                n += 1
        TotDihedralTypeClass = _np.delete(temp, range(n, m), axis=0)

        for mol in range(len(self.MOL)):
            IDs_dihedraltype = _np.zeros(self.nDIHEDRALtype[mol], dtype=object)
            dihedraltype = _np.zeros(self.nDIHEDRALtype[mol], dtype=int)
            dihedraltype_name = _np.zeros([self.nDIHEDRALtype[mol], 4], dtype='U5')
            if self.nDIHEDRALtype[mol] > 0:
                n = 0
                for i in range(len(TotDihedralTypeClass)):
                    IDsTemp = _np.zeros([0, 4], dtype=int)
                    indT1 = TotDihedralTypeClass[i, 0]
                    indT2 = TotDihedralTypeClass[i, 1]
                    indT3 = TotDihedralTypeClass[i, 2]
                    redundant = _np.all([indT1, indT2, indT3] == [indT3, indT2, indT1])
                    comp1 = TotDihedralType[mol] == [indT1, indT2, indT3]
                    comp2 = TotDihedralType[mol] == [indT3, indT2, indT1]
                    cmp1 = _np.all(comp1, axis=1)
                    cmp2 = _np.all(comp2, axis=1)
                    if _np.any(cmp1):
                        temp = TotDihedral[mol][cmp1]
                        IDsTemp = _np.concatenate([IDsTemp, temp], axis=0)
                        ind1 = temp[0, 0]
                        ind2 = temp[0, 1]
                        ind3 = temp[0, 2]
                        ind4 = temp[0, 3]
                        A = self.BEADtype[mol][ind1]
                        B = self.BEADtype[mol][ind2]
                        C = self.BEADtype[mol][ind3]
                        D = self.BEADtype[mol][ind4]
                        dihedraltype[n] = i
                        dihedraltype_name[n] = [A, B, C, D]
                        IDs_dihedraltype[n] = IDsTemp
                        n += 1
                    if _np.any(cmp2) and not redundant:
                        temp = TotDihedral[mol][cmp2]
                        IDsTemp = _np.concatenate([IDsTemp, temp], axis=0)
                        ind1 = temp[0, 0]
                        ind2 = temp[0, 1]
                        ind3 = temp[0, 2]
                        ind4 = temp[0, 3]
                        A = self.BEADtype[mol][ind1]
                        B = self.BEADtype[mol][ind2]
                        C = self.BEADtype[mol][ind3]
                        D = self.BEADtype[mol][ind4]
                        dihedraltype[n] = i
                        dihedraltype_name[n] = [A, B, C, D]
                        IDs_dihedraltype[n] = IDsTemp
                        n += 1
                    # IDsTemp = _np.delete(IDsTemp, 0, axis=0)
                    # IDs_dihedraltype[n] = IDsTemp
                    # if _np.any(cmp1) or _np.any(cmp2):
                    #     n += 1
                self.DIHEDRALtypeIdx[mol] = IDs_dihedraltype
                self.DIHEDRALtype[mol] = dihedraltype
                self.DIHEDRALtypeName[mol] = dihedraltype_name
            else:
                pass
                # print('There is no dihedral in the "' + self.MOL[mol] + '" molecule.')
#####################################################
#####################################################
class _cTRAJ:
    def __init__(self, inFile, mode):
        inFile = str(inFile)
        self._mode = str(mode)
        if inFile.endswith('.xtc'):
            self.Type = 'XTC'
        elif inFile.endswith('.gro'):
            self.Type = 'GRO'
        elif inFile.endswith('.pdb'):
            self.Type = 'PDB'
        elif inFile.endswith('.lammpstrj'):
            self.Type = 'LMP'
        elif inFile.endswith('.dcd'):
            self.Type = 'DCD'
        else:
            raise _scg4pyError('trajectory should be one of: ".xtc", ".dcd", ".gro", ".pdb", ".lammpstrj"')
        if self._mode == 'r':
            if self.Type == 'XTC' or self.Type == 'DCD':
                import mdtraj as _mdtraj
                self._fid = _mdtraj.open(inFile, mode='r')
            else:
                self._fid = open(inFile, mode='r')
        elif self._mode == 'w':
            if self.Type == 'XTC' or self.Type == 'DCD':
                import mdtraj as _mdtraj
                self._fid = _mdtraj.open(inFile, mode='w')
            else:
                self._fid = open(inFile, mode='w')
        else:
            raise _scg4pyError('opening mode should be one of "r" or "w".')
        self._setNone()

    def _setNone(self):
        self.time = None
        self.timeStep = None
        self.nAtom = None
        self.atId = None
        self.atName = None
        self.atType = None
        self.resId = None
        self.resName = None
        self.x = None
        self.y = None
        self.z = None
        self.xs = None
        self.ys = None
        self.zs = None
        self.xu = None
        self.yu = None
        self.zu = None
        self.ix = None
        self.iy = None
        self.iz = None
        self.fx = None
        self.fy = None
        self.fz = None
        self.lmpBox = None
        self.boxMat = None
        self.boxCryst = None
        self.eof = False

    def close(self):
        self._fid.close()

    def read(self, nAtom=None):
        self._setNone()
        if self.Type == 'GRO':
            self.readGROsnap(nAtom)
        elif self.Type == 'PDB':
            self.readPDBsnap(nAtom)
        elif self.Type == 'XTC':
            self.readXTCsnap(nAtom)
        elif self.Type == 'LMP':
            self.readLMPsnap(nAtom)
        elif self.Type == 'DCD':
            self.readDCDsnap(nAtom)

    def write(self, modelN=None, sysName=None):
        if self.Type == 'GRO':
            self.writeGROsnap()
        elif self.Type == 'PDB':
            self.writePDBsnap(modelN, sysName)
        elif self.Type == 'XTC':
            self.writeXTCsnap()
        elif self.Type == 'LMP':
            self.writeLMPsnap()
        elif self.Type == 'DCD':
            self.writeDCDsnap()

    def _box_mat2cryst(self):
        if self.boxMat is not None:
            lx, ly, lz, xy, xz, yz = self.boxMat[0], self.boxMat[1], self.boxMat[2], \
                                     self.boxMat[3], self.boxMat[4], self.boxMat[5]
            if lx == 0 or ly == 0 or lz == 0:
                self.boxCryst = _np.zeros(6, dtype=_np.float32)
            else:
                a = lx
                b = _np.sqrt(ly ** 2 + xy ** 2)
                c = _np.sqrt(lz ** 2 + xz ** 2 + yz ** 2)
                alpha = round(_np.rad2deg(_np.arccos((xy * xz + ly * yz) / (b * c))), 3)
                beta = round(_np.rad2deg(_np.arccos(xz / c)), 3)
                gamma = round(_np.rad2deg(_np.arccos(xy / b)), 3)
                self.boxCryst = _np.array([a, b, c, alpha, beta, gamma], dtype=_np.float32)
        else:
            self.boxCryst = None

    def _box_cryst2mat(self):
        if len(self.boxCryst) is not None:
            a, b, c, alpha, beta, gamma = self.boxCryst[0], self.boxCryst[1], self.boxCryst[2], \
                                          self.boxCryst[3], self.boxCryst[4], self.boxCryst[5]
            if a == 0 or b == 0 or c == 0:
                self.boxMat = _np.zeros(6, dtype=_np.float32)
            else:
                lx = a
                xy = b * _np.cos(_np.deg2rad(gamma))
                xz = c * _np.cos(_np.deg2rad(beta))
                ly = _np.sqrt(b ** 2 - xy ** 2)
                yz = (b * c * _np.cos(_np.deg2rad(alpha)) - xy * xz) / ly
                lz = _np.sqrt(c ** 2 - xz ** 2 - yz ** 2)
                self.boxMat = _np.array([lx, ly, lz, xy, xz, yz], dtype=_np.float32)
        else:
            self.boxMat = None

    def readGROsnap(self, nAtom=None):
        try:
            self.time = float(self._fid.readline().split('t=')[1].strip())
            self.nAtom = int(self._fid.readline().strip())
            if (nAtom is not None) and (self.nAtom != nAtom):
                self.close()
                raise _scg4pyError('Number of atoms in the structure and in the option file are not equal.')
            self.resId = _np.zeros(self.nAtom, dtype=_np.int32)
            self.resName = _np.zeros(self.nAtom, dtype='U5')
            self.atName = _np.zeros(self.nAtom, dtype='U5')
            self.atId = _np.zeros(self.nAtom, dtype=_np.int32)
            self.x = _np.zeros(self.nAtom, dtype=_np.float32)
            self.y = _np.zeros(self.nAtom, dtype=_np.float32)
            self.z = _np.zeros(self.nAtom, dtype=_np.float32)
            for i in _np.arange(self.nAtom):
                line = self._fid.readline()
                self.resName[i] = line[5:10].strip()
                self.atName[i] = line[10:15].strip()
                self.x[i] = float(line[20:28]) * 10
                self.y[i] = float(line[28:36]) * 10
                self.z[i] = float(line[36:44]) * 10
                if i == 0:
                    self.resId[i] = 1
                    prevId = int(line[0:5])
                else:
                    if int(line[0:5]) == prevId:
                        self.resId[i] = self.resId[i - 1]
                    else:
                        self.resId[i] = self.resId[i - 1] + 1
                        prevId = int(line[0:5])
            self.atId = _np.arange(1, self.nAtom + 1)
            VecComp = _np.array(self._fid.readline().split(), dtype=float)
            VecComp = VecComp * 10
            if _np.size(VecComp) == 3:
                lx = VecComp[0]
                ly = VecComp[1]
                lz = VecComp[2]
                xy = xz = yz = 0
            else:
                lx = VecComp[0]
                ly = VecComp[1]
                lz = VecComp[2]
                xy = VecComp[5]
                xz = VecComp[7]
                yz = VecComp[8]
            self.boxMat = _np.array([lx, ly, lz, xy, xz, yz], dtype=_np.float32)
            self._box_mat2cryst()
        except (IndexError, ValueError, TypeError, EOFError):
            self.eof = True

    def writeGROsnap(self):
        cond = len(self.resId) == len(self.resName) == len(self.atName) == len(self.atId) == \
               len(self.x) == len(self.y) == len(self.z) == self.nAtom
        if not cond:
            mess = '"gro" writing error: input variables are not compatible with each other.'
            raise _scg4pyError(mess)
        self._fid.write('Generated by SCG4PY program: t= {}\n'.format(self.time))
        self._fid.write('{:6d}\n'.format(self.nAtom))
        for i in _np.arange(len(self.atId)):
            newResId = int(self.resId[i] % 1e5)
            newAtId = int(self.atId[i] % 1e5)
            self._fid.write('{0:5d}{1:<5s}{2:^5s}{3:5d}'.format(newResId, self.resName[i], self.atName[i], newAtId))
            self._fid.write('{0:8.3f}{1:8.3f}{2:8.3f}\n'.format(self.x[i] / 10, self.y[i] / 10, self.z[i] / 10))
        self.boxMat = self.boxMat / 10
        lx, ly, lz, xy, xz, yz = self.boxMat[0], self.boxMat[1], self.boxMat[2], \
                                 self.boxMat[3], self.boxMat[4], self.boxMat[5]
        if xy == xz == yz == 0:
            self._fid.write('{0:10.5f}{1:10.5f}{2:10.5f}\n'.format(lx, ly, lz))
        else:
            self._fid.write(
                '{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}{6:10.5f}{7:10.5f}{8:10.5f}\n'.format(
                    lx, ly, lz, 0, 0, xy, 0, xz, yz))

    def readPDBsnap(self, nAtom=None):
        try:
            ftell = self._fid.tell()
            self.boxCryst = _np.zeros(6, dtype=_np.float32)
            self.time = 0.0
            line = self._fid.readline()
            while ('ATOM' not in line) and ('HETATM' not in line):
                if ('TITLE' in line) and ('t=' in line):
                    x = line.split('t=')[1]
                    if _isfloat(x):
                        self.time = float(x)
                    else:
                        self.time = 0.0
                elif 'CRYST1' in line:
                    a = float(line[6:15])
                    b = float(line[15:24])
                    c = float(line[24:33])
                    al = float(line[33:40])
                    bt = float(line[40:47])
                    gm = float(line[47:54])
                    self.boxCryst = _np.array([a, b, c, al, bt, gm], dtype=_np.float32)
                line = self._fid.readline()
                if self._fid.tell() != ftell:
                    ftell = self._fid.tell()
                else:
                    raise EOFError
            self._box_cryst2mat()
            if any([i == 0 for i in self.boxMat]):
                raise _scg4pyError('some snapshots do not have box information "CRYST1" section.')
            self.nAtom = nAtom
            self.resId = _np.zeros(self.nAtom, dtype=_np.int32)
            self.resName = _np.zeros(self.nAtom, dtype='U5')
            self.atName = _np.zeros(self.nAtom, dtype='U5')
            self.atId = _np.zeros(self.nAtom, dtype=_np.int32)
            self.x = _np.zeros(self.nAtom, dtype=_np.float32)
            self.y = _np.zeros(self.nAtom, dtype=_np.float32)
            self.z = _np.zeros(self.nAtom, dtype=_np.float32)
            i = 0
            while ('ENDMDL' not in line) and ('END' not in line):
                if ('ATOM' in line) or ('HETATM' in line):
                    if (nAtom is not None) and (self.nAtom != nAtom):
                        self.close()
                        raise _scg4pyError('Number of atoms in the trajectory and in the option file are not equal.')
                    self.atId[i] = i + 1
                    self.atName[i] = line[12:16].strip()
                    self.resName[i] = line[17:21].strip()
                    if i == 0:
                        self.resId[i] = 1
                        prevId = int(line[22:26])
                    else:
                        if int(line[22:26]) == prevId:
                            self.resId[i] = self.resId[i - 1]
                        else:
                            self.resId[i] = self.resId[i - 1] + 1
                            prevId = int(line[22:26])
                    self.x[i] = line[30:38]
                    self.y[i] = line[38:46]
                    self.z[i] = line[46:54]
                    i += 1
                line = self._fid.readline()
                if self._fid.tell() != ftell:
                    ftell = self._fid.tell()
                else:
                    raise EOFError
            if (nAtom is not None) and (i != self.nAtom):
                self.close()
                raise _scg4pyError('Number of atoms in the trajectory and in the option file are not equal.')
        except (IndexError, ValueError, TypeError, EOFError):
            self.eof = True

    def writePDBsnap(self, modelN=None, sysName=None):
        cond = len(self.resId) == len(self.resName) == len(self.atName) == len(self.atId) == \
               len(self.x) == len(self.y) == len(self.z) == self.nAtom
        if not cond:
            mess = '"pdb" writing error: input variables are not compatible with each other.'
            raise _scg4pyError(mess)
        if sysName is None:
            sysName = ''
        if modelN is None:
            modelN = 1
        if self.boxCryst is None:
            self._box_mat2cryst()
        self._fid.write('REMARK    GENERATED BY SCG4PY\n')
        self._fid.write('TITLE     {0:} t={1:}\n'.format(sysName, self.time))
        a, b, c, alpha, beta, gamma = self.boxCryst[0], self.boxCryst[1], self.boxCryst[2], \
                                      self.boxCryst[3], self.boxCryst[4], self.boxCryst[5]
        self._fid.write('{0:6s}{1:9.3f}{2:9.3f}{3:9.3f}{4:7.2f}{5:7.2f}{6:7.2f} {7:<11s}{8:4s}\n'.format(
            'CRYST1', a, b, c, alpha, beta, gamma, 'P 1', '1'))
        self._fid.write('{0:6}    {1:4d}\n'.format('MODEL', int(modelN % 1e4)))
        for i in _np.arange(self.nAtom):
            self._fid.write('{0:6s}{1:5d} {2:^4s} {3:<4s} {4:4d}    '.format(
                'ATOM', int(self.atId[i] % 1e5), self.atName[i], self.resName[i], int(self.resId[i] % 1e4)))
            self._fid.write(
                '{0:8.3f}{1:8.3f}{2:8.3f}                       C\n'.format(self.x[i], self.y[i], self.z[i]))
        self._fid.write('TER\nENDMDL\n')

    def readXTCsnap(self, nAtom=None):
        try:
            snap = self._fid.read(1)
            if len(snap[1]) != 0:
                self.x = snap[0][0][:, 0] * 10
                self.y = snap[0][0][:, 1] * 10
                self.z = snap[0][0][:, 2] * 10
                self.nAtom = len(self.x)
                if (nAtom is not None) and (self.nAtom != nAtom):
                    self.close()
                    raise _scg4pyError('Number of atoms in the trajectory and in the option file are not equal.')
                self.time = snap[1][0]
                self.timeStep = snap[2][0]
                lx = snap[3][0][0][0] * 10
                ly = snap[3][0][1][1] * 10
                lz = snap[3][0][2][2] * 10
                xy = snap[3][0][1][0] * 10
                xz = snap[3][0][2][0] * 10
                yz = snap[3][0][2][1] * 10
                self.boxMat = _np.array([lx, ly, lz, xy, xz, yz], dtype=_np.float32)
                self._box_mat2cryst()
            else:
                self.eof = True
        except:
            self.eof = True

    def writeXTCsnap(self):
        if len(self.x) == len(self.y) == len(self.z):
            pass
        else:
            mess = '"xtc" writing error: input variables are not compatible with each other.'
            raise _scg4pyError(mess)
        lx, ly, lz, xy, xz, yz = self.boxMat[0], self.boxMat[1], self.boxMat[2], \
                                 self.boxMat[3], self.boxMat[4], self.boxMat[5]
        time = _np.zeros(1, dtype=_np.float32)
        step = _np.zeros(1, dtype=_np.int32)
        box = _np.zeros([1, 3, 3], dtype=_np.float32)
        xyz = _np.zeros([1, len(self.x), 3], dtype=_np.float32)
        xyz[0][:, 0] = self.x / 10
        xyz[0][:, 1] = self.y / 10
        xyz[0][:, 2] = self.z / 10
        time[0] = self.time
        step[0] = self.timeStep
        box[0][0][0] = lx / 10
        box[0][1][1] = ly / 10
        box[0][2][2] = lz / 10
        box[0][1][0] = xy / 10
        box[0][2][0] = xz / 10
        box[0][2][1] = yz / 10
        self._fid.write(xyz=xyz, time=time, step=step, box=box)

    def readDCDsnap(self, nAtom=None):
        try:
            snap = self._fid.read(1)
            if len(snap[0]) != 0:
                self.x = snap[0][0][:, 0]
                self.y = snap[0][0][:, 1]
                self.z = snap[0][0][:, 2]
                self.nAtom = len(self.x)
                if (nAtom is not None) and (self.nAtom != nAtom):
                    self.close()
                    raise _scg4pyError('Number of atoms in the trajectory and in the option file are not equal.')
                a = snap[1][0][0]
                b = snap[1][0][1]
                c = snap[1][0][2]
                al = snap[2][0][0]
                bt = snap[2][0][1]
                gm = snap[2][0][2]
                self.boxCryst = _np.array([a, b, c, al, bt, gm], dtype=_np.float32)
                self._box_cryst2mat()
            else:
                self.eof = True
        except:
            self.eof = True

    def writeDCDsnap(self):
        if len(self.x) == len(self.y) == len(self.z):
            pass
        else:
            mess = '"dcd" writing error: input variables are not compatible with each other.'
            raise _scg4pyError(mess)
        self._box_mat2cryst()
        a, b, c, al, bt, gm = self.boxCryst[0], self.boxCryst[1], self.boxCryst[2], \
                              self.boxCryst[3], self.boxCryst[4], self.boxCryst[5]
        boxLen = _np.zeros([1, 3], dtype=_np.float32)
        boxAng = _np.zeros([1, 3], dtype=_np.float32)
        xyz = _np.zeros([1, len(self.x), 3], dtype=_np.float32)
        xyz[0][:, 0] = self.x
        xyz[0][:, 1] = self.y
        xyz[0][:, 2] = self.z
        boxLen[0] = [a, b, c]
        boxAng[0] = [al, bt, gm]
        self._fid.write(xyz=xyz, cell_lengths=boxLen, cell_angles=boxAng)

    def readLMPsnap(self, nAtom=None):
        try:
            self._fid.readline()
            self.timeStep = int(self._fid.readline())
            self._fid.readline()
            self.nAtom = int(self._fid.readline())
            if (nAtom is not None) and (self.nAtom != nAtom):
                self.close()
                raise _scg4pyError('Number of atoms in the trajectory and in the option file are not equal.')
            lmpBox = {'xlo_bound': 0, 'xhi_bound': 0, 'xlo': 0, 'xhi': 0, 'ylo_bound': 0, 'yhi_bound': 0, 'ylo': 0,
                      'yhi': 0, 'zlo_bound': 0, 'zhi_bound': 0, 'zlo': 0, 'zhi': 0, 'xy': 0, 'xz': 0, 'yz': 0}
            if 'xy' in self._fid.readline():
                line = self._fid.readline().split()
                lmpBox['xlo_bound'] = float(line[0])
                lmpBox['xhi_bound'] = float(line[1])
                lmpBox['xy'] = float(line[2])
                line = self._fid.readline().split()
                lmpBox['ylo_bound'] = float(line[0])
                lmpBox['yhi_bound'] = float(line[1])
                lmpBox['xz'] = float(line[2])
                line = self._fid.readline().split()
                lmpBox['zlo_bound'] = float(line[0])
                lmpBox['zhi_bound'] = float(line[1])
                lmpBox['yz'] = float(line[2])
            else:
                line = self._fid.readline().split()
                lmpBox['xlo_bound'] = float(line[0])
                lmpBox['xhi_bound'] = float(line[1])
                lmpBox['xy'] = 0.0
                line = self._fid.readline().split()
                lmpBox['ylo_bound'] = float(line[0])
                lmpBox['yhi_bound'] = float(line[1])
                lmpBox['xz'] = 0.0
                line = self._fid.readline().split()
                lmpBox['zlo_bound'] = float(line[0])
                lmpBox['zhi_bound'] = float(line[1])
                lmpBox['yz'] = 0.0
            lmpBox['xlo'] = lmpBox['xlo_bound'] - min(0, lmpBox['xy'], lmpBox['xz'], (lmpBox['xy'] + lmpBox['xz']))
            lmpBox['xhi'] = lmpBox['xhi_bound'] - max(0, lmpBox['xy'], lmpBox['xz'], (lmpBox['xy'] + lmpBox['xz']))
            lmpBox['ylo'] = lmpBox['ylo_bound'] - min(0, lmpBox['yz'])
            lmpBox['yhi'] = lmpBox['yhi_bound'] - max(0, lmpBox['yz'])
            lmpBox['zlo'] = lmpBox['zlo_bound']
            lmpBox['zhi'] = lmpBox['zhi_bound']
            lx = lmpBox['xhi'] - lmpBox['xlo']
            ly = lmpBox['yhi'] - lmpBox['ylo']
            lz = lmpBox['zhi'] - lmpBox['zlo']
            xy, xz, yz = lmpBox['xy'], lmpBox['xz'], lmpBox['yz']
            self.boxMat = _np.array([lx, ly, lz, xy, xz, yz], dtype=_np.float32)
            self._box_mat2cryst()
            line = self._fid.readline().replace('ITEM: ATOMS', '')
            lsp = _np.array(line.split(), dtype='U6')
            idI = typeI = xuI = yuI = zuI = xI = yI = zI = xsI = ysI = zsI = None
            ixI = iyI = izI = fxI = fyI = fzI = None
            if 'id' in lsp:
                self.atId = _np.zeros(self.nAtom, dtype=_np.int32)
                idI = _np.where(lsp == 'id')[0][0]
            if 'type' in lsp:
                self.atType = _np.zeros(self.nAtom, dtype=_np.int32)
                typeI = _np.where(lsp == 'type')[0][0]
            if 'ix' in lsp:
                self.ix = _np.zeros(self.nAtom, dtype=_np.int32)
                ixI = _np.where(lsp == 'ix')[0][0]
            if 'iy' in lsp:
                self.iy = _np.zeros(self.nAtom, dtype=_np.int32)
                iyI = _np.where(lsp == 'iy')[0][0]
            if 'iz' in lsp:
                self.iz = _np.zeros(self.nAtom, dtype=_np.int32)
                izI = _np.where(lsp == 'iz')[0][0]
            if 'x' in lsp:
                self.x = _np.zeros(self.nAtom, dtype=_np.float32)
                xI = _np.where(lsp == 'x')[0][0]
            if 'y' in lsp:
                self.y = _np.zeros(self.nAtom, dtype=_np.float32)
                yI = _np.where(lsp == 'y')[0][0]
            if 'z' in lsp:
                self.z = _np.zeros(self.nAtom, dtype=_np.float32)
                zI = _np.where(lsp == 'z')[0][0]
            if 'xs' in lsp:
                self.xs = _np.zeros(self.nAtom, dtype=_np.float32)
                xsI = _np.where(lsp == 'xs')[0][0]
            if 'ys' in lsp:
                self.ys = _np.zeros(self.nAtom, dtype=_np.float32)
                ysI = _np.where(lsp == 'ys')[0][0]
            if 'zs' in lsp:
                self.zs = _np.zeros(self.nAtom, dtype=_np.float32)
                zsI = _np.where(lsp == 'zs')[0][0]
            if 'xu' in lsp:
                self.xu = _np.zeros(self.nAtom, dtype=_np.float32)
                xuI = _np.where(lsp == 'xu')[0][0]
            if 'yu' in lsp:
                self.yu = _np.zeros(self.nAtom, dtype=_np.float32)
                yuI = _np.where(lsp == 'yu')[0][0]
            if 'zu' in lsp:
                self.zu = _np.zeros(self.nAtom, dtype=_np.float32)
                zuI = _np.where(lsp == 'zu')[0][0]
            if 'fx' in lsp:
                self.fx = _np.zeros(self.nAtom, dtype=_np.float32)
                fxI = _np.where(lsp == 'fx')[0][0]
            if 'fy' in lsp:
                self.fy = _np.zeros(self.nAtom, dtype=_np.float32)
                fyI = _np.where(lsp == 'fy')[0][0]
            if 'fz' in lsp:
                self.fz = _np.zeros(self.nAtom, dtype=_np.float32)
                fzI = _np.where(lsp == 'fz')[0][0]
            for i in _np.arange(self.nAtom):
                lsp = self._fid.readline().split()
                if idI is not None:
                    self.atId[i] = lsp[idI]
                if typeI is not None:
                    self.atType[i] = lsp[typeI]
                if xI is not None:
                    self.x[i] = lsp[xI]
                if yI is not None:
                    self.y[i] = lsp[yI]
                if zI is not None:
                    self.z[i] = lsp[zI]
                if xsI is not None:
                    self.xs[i] = lsp[xsI]
                if ysI is not None:
                    self.ys[i] = lsp[ysI]
                if zsI is not None:
                    self.zs[i] = lsp[zsI]
                if xuI is not None:
                    self.xu[i] = lsp[xuI]
                if yuI is not None:
                    self.yu[i] = lsp[yuI]
                if zuI is not None:
                    self.zu[i] = lsp[zuI]
                if ixI is not None:
                    self.ix[i] = lsp[ixI]
                if iyI is not None:
                    self.iy[i] = lsp[iyI]
                if izI is not None:
                    self.iz[i] = lsp[izI]
                if fxI is not None:
                    self.fx[i] = lsp[fxI]
                if fyI is not None:
                    self.fy[i] = lsp[fyI]
                if fzI is not None:
                    self.fz[i] = lsp[fzI]
            if idI is not None:
                argS = _np.argsort(self.atId)
                self.atId = self.atId[argS]
                if typeI is not None:
                    self.atType = self.atId[argS]
                if xI is not None:
                    self.x = self.x[argS]
                if yI is not None:
                    self.y = self.y[argS]
                if zI is not None:
                    self.z = self.z[argS]
                if xsI is not None:
                    self.xs = self.xs[argS]
                if ysI is not None:
                    self.ys = self.ys[argS]
                if zsI is not None:
                    self.zs = self.zs[argS]
                if xuI is not None:
                    self.xu = self.xu[argS]
                if yuI is not None:
                    self.yu = self.yu[argS]
                if zuI is not None:
                    self.zu = self.zu[argS]
                if ixI is not None:
                    self.ix = self.ix[argS]
                if iyI is not None:
                    self.iy = self.iy[argS]
                if izI is not None:
                    self.iz = self.iz[argS]
                if fxI is not None:
                    self.fx = self.fx[argS]
                if fyI is not None:
                    self.fy = self.fy[argS]
                if fzI is not None:
                    self.fz = self.fz[argS]
        except (IndexError, ValueError, TypeError, EOFError):
            self.eof = True

    def unscaleSnap(self):
        cond = (self.xs is not None) and (self.ys is not None) and (self.zs is not None)
        if cond:
            xs, ys, zs = self.xs, self.ys, self.zs
        else:
            mess = 'Unscaling error: There are no scaled coordinates.'
            raise _scg4pyError(mess)
        if self.lmpBox is None:
            xlo, ylo, zlo = 0.0, 0.0, 0.0
        lx, ly, lz, xy, xz, yz = self.boxMat[0], self.boxMat[1], self.boxMat[2], \
                                 self.boxMat[3], self.boxMat[4], self.boxMat[5]
        x = xlo + xs * lx + ys * xy + zs * xz
        y = ylo + ys * ly + zs * yz
        z = zlo + zs * lz
        self.x, self.y, self.z = x, y, z

    def scaleSnap(self):
        cond1 = (self.x is not None) and (self.y is not None) and (self.z is not None)
        cond2 = (self.xu is not None) and (self.yu is not None) and (self.zu is not None)
        if cond1:
            x, y, z = self.x, self.y, self.z
        elif cond2:
            x, y, z = self.xu, self.yu, self.zu
        else:
            mess = 'Scaling error: There are no unscaled coordinates.'
            raise _scg4pyError(mess)
        if self.lmpBox is None:
            xlo, ylo, zlo = 0.0, 0.0, 0.0
        lx, ly, lz, xy, xz, yz = self.boxMat[0], self.boxMat[1], self.boxMat[2], \
                                 self.boxMat[3], self.boxMat[4], self.boxMat[5]
        xs = (x - xlo) / lx - (y - ylo) * xy / (lx * ly) + (z - zlo) * (yz * xy - xz * ly) / (lx * ly * lz)
        ys = (y - ylo) / ly - yz * (z - zlo) / (ly * lz)
        zs = (z - zlo) / lz
        self.xs, self.ys, self.zs = xs, ys, zs

    def unwrapLMPsnap(self):
        cond = (self.xs is not None) and (self.ys is not None) and (self.zs is not None)
        cond = cond and (self.ix is not None) and (self.iy is not None) and (self.iz is not None)
        if cond:
            pass
        else:
            mess = 'Unwrapping error: There are no "xs", "ys", "zs", "ix", "iy", and iz" entries.'
            raise _scg4pyError(mess)
        if self.lmpBox is None:
            xlo, ylo, zlo = 0.0, 0.0, 0.0
        lx, ly, lz, xy, xz, yz = self.boxMat[0], self.boxMat[1], self.boxMat[2], \
                                 self.boxMat[3], self.boxMat[4], self.boxMat[5]
        xus = self.xs + self.ix
        yus = self.ys + self.iy
        zus = self.zs + self.iz
        xu = xlo + xus * lx + yus * xy + zus * xz
        yu = ylo + yus * ly + zus * yz
        zu = zlo + zus * lz
        self.xu, self.yu, self.zu = xu, yu, zu

    def wrapLMPsnap(self):
        if (self.xu is None) or (self.yu is None) or (self.zu is None):
            mess = 'Unwrapping error: There are no "xs", "ys", "zs", "ix", "iy", and iz" entries.'
            raise _scg4pyError(mess)
        self.scaleSnap()
        self.ix = _np.floor(self.xs)
        self.iy = _np.floor(self.ys)
        self.iz = _np.floor(self.zs)
        self.xs = self.xs - self.ix
        self.ys = self.ys - self.iy
        self.zs = self.zs - self.iz

    def writeLMPsnap(self):
        self._fid.write('ITEM: TIMESTEP\n{0:d}\n'.format(self.timeStep))
        self._fid.write('ITEM: NUMBER OF ATOMS\n{0:d}\n'.format(self.nAtom))
        lx, ly, lz, xy, xz, yz = self.boxMat[0], self.boxMat[1], self.boxMat[2], \
                                 self.boxMat[3], self.boxMat[4], self.boxMat[5]
        xlo_bound = min(0, xy, xz, (xy + xz))
        xhi_bound = lx + max(0, xy, xz, (xy + xz))
        ylo_bound = min(0, yz)
        yhi_bound = ly + max(0, yz)
        zlo_bound = 0.0
        zhi_bound = lz
        if xy == xz == yz == 0:
            self._fid.write('ITEM: BOX BOUNDS pp pp pp\n')
            self._fid.write('{0:.5f} {1:.5f}\n{2:.5f} {3:.5f}\n{4:.5f} {5:.5f}\n'.format(xlo_bound, xhi_bound,
                                                                                         ylo_bound, yhi_bound,
                                                                                         zlo_bound, zhi_bound))
        else:
            self._fid.write('ITEM: BOX BOUNDS xy xz yz\n')
            self._fid.write('{0:.5f} {1:.5f} {2:.5f}\n{3:.5f} {4:.5f} {5:.5f}\n{6:.5f} {7:.5f} {8:.5f}\n'.format(
                xlo_bound, xhi_bound, xy, ylo_bound, yhi_bound, xz, zlo_bound, zhi_bound, yz))
        self._fid.write('ITEM: ATOMS')
        if self.atId is not None:
            self._fid.write(' id')
        if self.atType is not None:
            self._fid.write(' type')
        if self.ix is not None:
            self._fid.write(' ix')
        if self.iy is not None:
            self._fid.write(' iy')
        if self.iz is not None:
            self._fid.write(' iz')
        if self.x is not None:
            self._fid.write(' x')
        if self.y is not None:
            self._fid.write(' y')
        if self.z is not None:
            self._fid.write(' z')
        if self.xs is not None:
            self._fid.write(' xs')
        if self.ys is not None:
            self._fid.write(' ys')
        if self.zs is not None:
            self._fid.write(' zs')
        if self.xu is not None:
            self._fid.write(' xu')
        if self.yu is not None:
            self._fid.write(' yu')
        if self.zu is not None:
            self._fid.write(' zu')
        if self.fx is not None:
            self._fid.write(' fx')
        if self.fy is not None:
            self._fid.write(' fy')
        if self.fz is not None:
            self._fid.write(' fz')
        self._fid.write('\n')
        for i in _np.arange(self.nAtom):
            line = ''
            if self.atId is not None:
                line += ' {0:d}'.format(self.atId[i])
            if self.atType is not None:
                line += ' {0:d}'.format(self.atType[i])
            if self.ix is not None:
                line += ' {0:d}'.format(self.ix[i])
            if self.iy is not None:
                line += ' {0:d}'.format(self.iy[i])
            if self.iz is not None:
                line += ' {0:d}'.format(self.iz[i])
            if self.x is not None:
                line += ' {0:.3f}'.format(self.x[i])
            if self.y is not None:
                line += ' {0:.3f}'.format(self.y[i])
            if self.z is not None:
                line += ' {0:.3f}'.format(self.z[i])
            if self.xs is not None:
                line += ' {0:.3f}'.format(self.xs[i])
            if self.ys is not None:
                line += ' {0:.3f}'.format(self.ys[i])
            if self.zs is not None:
                line += ' {0:.3f}'.format(self.zs[i])
            if self.xu is not None:
                line += ' {0:.3f}'.format(self.xu[i])
            if self.yu is not None:
                line += ' {0:.3f}'.format(self.yu[i])
            if self.zu is not None:
                line += ' {0:.3f}'.format(self.zu[i])
            if self.fx is not None:
                line += ' {0:.3f}'.format(self.fx[i])
            if self.fy is not None:
                line += ' {0:.3f}'.format(self.fy[i])
            if self.fz is not None:
                line += ' {0:.3f}'.format(self.fz[i])
            line = line.strip()
            self._fid.write(line + '\n')
        self._setNone()

    def rmPBC(self, MolId, BondsMap):
        if self.Type is not 'LMP':
            self.scaleSnap()
        else:
            if self.xs is not None and self.ys is not None and self.zs is not None:
                pass
            elif self.x is not None and self.y is not None and self.z is not None:
                self.scaleSnap()
        lx, ly, lz, xy, xz, yz = self.boxMat[0], self.boxMat[1], self.boxMat[2], \
                                 self.boxMat[3], self.boxMat[4], self.boxMat[5]
        xs = _np.copy(self.xs)
        ys = _np.copy(self.ys)
        zs = _np.copy(self.zs)
        dx = 10.0 / lx
        dy = 10.0 / ly
        dz = 10.0 / lz
        xl_id = xs <= dx
        xr_id = xs >= (1 - dx)
        yl_id = ys <= dy
        yr_id = ys >= (1 - dy)
        zl_id = zs <= dz
        zr_id = zs >= (1 - dz)
        xl_molid = _np.unique(MolId[xl_id])
        xr_molid = _np.unique(MolId[xr_id])
        yl_molid = _np.unique(MolId[yl_id])
        yr_molid = _np.unique(MolId[yr_id])
        zl_molid = _np.unique(MolId[zl_id])
        zr_molid = _np.unique(MolId[zr_id])
        xyz_molid = _np.concatenate([xl_molid, xr_molid, yl_molid, yr_molid, zl_molid, zr_molid])
        uniq, counts = _np.unique(xyz_molid, return_counts=True)
        pbc_molid = uniq[counts > 1]
        ### minimum image calculation
        atId = 0
        n = 0
        while n < len(pbc_molid) and atId < len(xs):
            if pbc_molid[n] == MolId[atId]:
                xA = xs[atId]
                yA = ys[atId]
                zA = zs[atId]
                if BondsMap[atId] is not None:
                    B_list = BondsMap[atId]
                    xB = xs[B_list]
                    yB = ys[B_list]
                    zB = zs[B_list]
                    xB_mi = xB - _np.rint(xB - xA)
                    yB_mi = yB - _np.rint(yB - yA)
                    zB_mi = zB - _np.rint(zB - zA)
                    xs[B_list] = xB_mi
                    ys[B_list] = yB_mi
                    zs[B_list] = zB_mi
                atId += 1
                if atId < len(xs) and pbc_molid[n] != MolId[atId]:
                    n += 1
            else:
                atId += 1
        ### unscaling
        x = xs * lx + ys * xy + zs * xz
        y = ys * ly + zs * yz
        z = zs * lz
        return xs, ys, zs, x, y, z
#####################################################
#####################################################
class _cTOP:
    def __init__(self, inFile):
        fid = open(inFile, 'r')
        mapFile = fid.readlines()
        fid.close()
        for line in mapFile:
            if 'MOL' in line.split('#')[0].upper():
                lsplit0 = line.split('#')[0].split('=')[0].strip()
                lsplit1 = line.split('#')[0].split('=')[1].strip()
                if lsplit0.upper() == 'MOL':
                    self.MOL = _np.array(lsplit1.split(), dtype='<U5')
                elif lsplit0.upper() == 'NMOL':
                    self.nMOL = _np.array(lsplit1.split(), dtype=int)
            if 'EXCL' in line.split('#')[0].upper():
                self.EXCL = int(line.split('#')[0].split('=')[1].strip())
        if len(self.MOL) > len(_np.unique(self.MOL)):
            mes = 'there is a redundancy in the molecules name.'
            raise _scg4pyError(mes)
        if len(self.MOL) != len(self.nMOL):
            raise ValueError('The length of "MOL" and "NMOL" entry should be equal.\n')
        self.nBEAD = _np.zeros(len(self.MOL), dtype=int)
        self.BEADname = _np.zeros(len(self.MOL), dtype=object)
        self.BEADtype = _np.zeros(len(self.MOL), dtype=object)
        self.BEADtypeSet = _np.zeros(0, dtype='U5')
        self.BEADtypeSetIdx = _np.zeros(0, dtype=object)
        self.LMPtype = _np.zeros(len(self.MOL), dtype=object)
        self.BEADch = _np.zeros(len(self.MOL), dtype=object)
        self.BEADms = _np.zeros(len(self.MOL), dtype=object)
        self.nBONDtype = _np.zeros(len(self.MOL), dtype=int)
        self.BONDtype = _np.zeros(len(self.MOL), dtype=object)
        self.BONDtypeName = _np.zeros(len(self.MOL), dtype=object)
        self.BONDtypeIdx = _np.zeros(len(self.MOL), dtype=object)
        self.BONDtypeSet = _np.zeros(0, dtype=int)
        self.BONDtypeSetName = _np.zeros([0, 2], dtype='U5')
        self.BONDtypeSetIdx = _np.zeros(0, dtype=object)
        self.nANGLEtype = _np.zeros(len(self.MOL), dtype=int)
        self.ANGLEtype = _np.zeros(len(self.MOL), dtype=object)
        self.ANGLEtypeName = _np.zeros(len(self.MOL), dtype=object)
        self.ANGLEtypeIdx = _np.zeros(len(self.MOL), dtype=object)
        self.ANGLEtypeSet = _np.zeros(0, dtype=int)
        self.ANGLEtypeSetName = _np.zeros([0, 3], dtype='U5')
        self.ANGLEtypeSetIdx = _np.zeros(0, dtype=object)
        self.nDIHEDRALtype = _np.zeros(len(self.MOL), dtype=int)
        self.DIHEDRALtype = _np.zeros(len(self.MOL), dtype=object)
        self.DIHEDRALtypeName = _np.zeros(len(self.MOL), dtype=object)
        self.DIHEDRALtypeIdx = _np.zeros(len(self.MOL), dtype=object)
        self.DIHEDRALtypeSet = _np.zeros(0, dtype=int)
        self.DIHEDRALtypeSetName = _np.zeros([0, 3], dtype=int)
        self.DIHEDRALtypeSetIdx = _np.zeros(0, dtype=object)
        self.NonBONDED_Set = _np.zeros([0, 2], dtype='U5')
        self.NonBONDED_SetIdx = _np.zeros(0, dtype=object)
        self.Excl_Mol = []
        for mol in range(len(self.MOL)):
            line = 0
            while line < len(mapFile):
                lsplit0 = mapFile[line].split('#')[0].split('=')[0].strip().split(':')
                if len(lsplit0) == 3:
                    lsplit1 = mapFile[line].split('#')[0].split('=')[1].strip().upper()
                    lsplit0[0] = lsplit0[0].upper()
                    lsplit0[2] = lsplit0[2].upper()
                    if lsplit0[0] == 'MOL' and (lsplit0[2] == 'BEAD' or lsplit0[2] == 'BOND' or
                                                lsplit0[2] == 'ANGLE' or lsplit0[2] == 'DIHEDRAL'):
                        if lsplit0[1] in self.MOL:
                            pass
                        else:
                            mes = 'error in line {}.'.format(line + 1)
                            raise _scg4pyError(mes)
                    if lsplit0[0] == 'MOL' and lsplit0[1] == self.MOL[mol] and lsplit0[2] == 'BEAD':
                        nbead = int(lsplit1)
                        n = 0
                        line += 1
                        bead_name = _np.zeros(nbead, dtype='<U5')
                        bead_type = _np.zeros(nbead, dtype='<U5')
                        lmpType = _np.zeros(nbead, dtype=int)
                        bead_ch = _np.zeros(nbead, dtype=float)
                        bead_ms = _np.zeros(nbead, dtype=float)
                        while n < nbead:
                            lco = mapFile[line].split('#')[0]
                            if 'MOL' in lco.upper():
                                raise TypeError('Error in "MOL:{}:BEAD" section'.format(self.MOL[mol]))
                            elif len(lco.split()) == 6:
                                bead_name[n] = lco.split()[1]
                                bead_type[n] = lco.split()[2]
                                lmpType[n] = lco.split()[3]
                                bead_ch[n] = lco.split()[4]
                                bead_ms[n] = lco.split()[5]
                                n += 1
                                line += 1
                            else:
                                line += 1
                        self.nBEAD[mol] = nbead
                        self.BEADname[mol] = bead_name
                        self.BEADtype[mol] = bead_type
                        self.LMPtype[mol] = lmpType
                        self.BEADch[mol] = bead_ch
                        self.BEADms[mol] = bead_ms
                    elif lsplit0[0] == 'MOL' and lsplit0[1] == self.MOL[mol] and lsplit0[2] == 'BOND':
                        nbond_type = int(lsplit1)
                        n = 0
                        line += 1
                        bond_type = _np.zeros(nbond_type, dtype=int)
                        bond_type_name = _np.zeros([nbond_type, 2], dtype='<U5')
                        bond_type_idx = _np.zeros(nbond_type, dtype=object)
                        while n < nbond_type:
                            lco = mapFile[line].split('#')[0]
                            if 'MOL' in lco:
                                raise TypeError('Error in "MOL:{}:BOND" section'.format(self.MOL[mol]))
                            temp = lco.split(':')[0].split()
                            try:
                                bond_type[n] = int(temp[0])
                                A = temp[1].split('-')[0].strip()
                                B = temp[1].split('-')[1].strip()
                                bond_type_name[n] = [A, B]
                                temp = lco.split(':')[1].split(',')
                                temp = [i.split() for i in temp if bool(i.split())]
                                bond_type_idx[n] = _np.array(temp, dtype=int)
                                n += 1
                            except:
                                pass
                            line += 1
                        self.nBONDtype[mol] = nbond_type
                        self.BONDtype[mol] = _np.zeros(0, dtype=int)
                        self.BONDtypeIdx[mol] = _np.zeros(0, dtype=object)
                        self.BONDtypeName[mol] = _np.zeros([0, 2], dtype='<U5')
                        if nbond_type > 0:
                            self.BONDtype[mol] = bond_type - 1
                            self.BONDtypeIdx[mol] = bond_type_idx - 1
                            self.BONDtypeName[mol] = bond_type_name
                    elif lsplit0[0] == 'MOL' and lsplit0[1] == self.MOL[mol] and lsplit0[2] == 'ANGLE':
                        nangle_type = int(lsplit1)
                        n = 0
                        line += 1
                        angle_type = _np.zeros(nangle_type, dtype=int)
                        angle_type_name = _np.zeros([nangle_type, 3], dtype='<U5')
                        angle_type_idx = _np.zeros(nangle_type, dtype=object)
                        while n < nangle_type:
                            lco = mapFile[line].split('#')[0]
                            if 'MOL' in lco:
                                raise TypeError('Error in "MOL:{}:BOND" section'.format(self.MOL[mol]))
                            temp = lco.split(':')[0].split()
                            try:
                                angle_type[n] = int(temp[0])
                                A = temp[1].split('-')[0].strip()
                                B = temp[1].split('-')[1].strip()
                                C = temp[1].split('-')[2].strip()
                                angle_type_name[n] = [A, B, C]
                                temp = lco.split(':')[1].split(',')
                                temp = [i.split() for i in temp if bool(i.split())]
                                angle_type_idx[n] = _np.array(temp, dtype=int)
                                n += 1
                            except:
                                pass
                            line += 1
                        self.nANGLEtype[mol] = nangle_type
                        self.ANGLEtype[mol] = _np.zeros(0, dtype=int)
                        self.ANGLEtypeIdx[mol] = _np.zeros(0, dtype=object)
                        self.ANGLEtypeName[mol] = _np.zeros([0, 3], dtype='<U5')
                        if nangle_type > 0:
                            self.ANGLEtype[mol] = angle_type - 1
                            self.ANGLEtypeIdx[mol] = angle_type_idx - 1
                            self.ANGLEtypeName[mol] = angle_type_name
                    elif lsplit0[0] == 'MOL' and lsplit0[1] == self.MOL[mol] and lsplit0[2] == 'DIHEDRAL':
                        ndihedral_type = int(lsplit1)
                        n = 0
                        line += 1
                        dihedral_type = _np.zeros(ndihedral_type, dtype=int)
                        dihedral_type_name = _np.zeros([ndihedral_type, 4], dtype='<U5')
                        dihedral_type_idx = _np.zeros(ndihedral_type, dtype=object)
                        while n < ndihedral_type:
                            lco = mapFile[line].split('#')[0]
                            if 'MOL' in lco:
                                raise TypeError('Error in "MOL:{}:BOND" section'.format(self.MOL[mol]))
                            temp = lco.split(':')[0].split()
                            try:
                                dihedral_type[n] = int(temp[0])
                                A = temp[1].split('-')[0].strip()
                                B = temp[1].split('-')[1].strip()
                                C = temp[1].split('-')[2].strip()
                                D = temp[1].split('-')[3].strip()
                                dihedral_type_name[n] = [A, B, C, D]
                                temp = lco.split(':')[1].split(',')
                                temp = [i.split() for i in temp if bool(i.split())]
                                dihedral_type_idx[n] = _np.array(temp, dtype=int)
                                n += 1
                            except:
                                pass
                            line += 1
                        self.nDIHEDRALtype[mol] = ndihedral_type
                        self.DIHEDRALtype[mol] = _np.zeros(0, dtype=int)
                        self.DIHEDRALtypeIdx[mol] = _np.zeros(0, dtype=object)
                        self.DIHEDRALtypeName[mol] = _np.zeros([0, 4], dtype='<U5')
                        if ndihedral_type > 0:
                            self.DIHEDRALtype[mol] = dihedral_type - 1
                            self.DIHEDRALtypeIdx[mol] = dihedral_type_idx - 1
                            self.DIHEDRALtypeName[mol] = dihedral_type_name
                line += 1
        self.MOLnum = _np.zeros(_np.sum(self.nMOL * self.nBEAD), dtype=int)
        n = 0
        Mn = 0
        for mol in range(len(self.MOL)):
            for i in range(self.nMOL[mol]):
                self.MOLnum[n: n + self.nBEAD[mol]] = Mn
                n += self.nBEAD[mol]
                Mn += 1

    ##########################################################

    def SetBondedIndex(self):
        BeadCumSum = _np.append([0], _np.cumsum(self.nBEAD * self.nMOL))
        BONDtypeIdx = _np.copy(self.BONDtypeIdx)
        ANGLEtypeIdx = _np.copy(self.ANGLEtypeIdx)
        DIHEDRALtypeIdx = _np.copy(self.DIHEDRALtypeIdx)
        Excl_Mol = _np.zeros(sum(self.nMOL), object)
        nM = 0
        for mol in range(len(self.MOL)):
            Bcat = _np.zeros([0, 2], dtype=int)
            Acat = _np.zeros([0, 2], dtype=int)
            Dcat = _np.zeros([0, 2], dtype=int)
            excl = None
            if self.nBONDtype[mol] > 0:
                BONDtypeIdx[mol] = self.BONDtypeIdx[mol] + BeadCumSum[mol]
                Bcat = _np.concatenate(BONDtypeIdx[mol][:])
            if self.nANGLEtype[mol] > 0:
                ANGLEtypeIdx[mol] = self.ANGLEtypeIdx[mol] + BeadCumSum[mol]
                Acat = _np.concatenate(ANGLEtypeIdx[mol][:])[:, [0, 2]]
            if self.nDIHEDRALtype[mol] > 0:
                DIHEDRALtypeIdx[mol] = self.DIHEDRALtypeIdx[mol] + BeadCumSum[mol]
                Dcat = _np.concatenate(DIHEDRALtypeIdx[mol][:])[:, [0, 3]]
            if self.EXCL == 3:
                excl = _np.concatenate([Bcat, Acat, Dcat])
            elif self.EXCL == 2:
                excl = _np.concatenate([Bcat, Acat])
            elif self.EXCL == 1:
                excl = Bcat
            if excl is not None:
                for i in range(self.nMOL[mol]):
                    Excl_Mol[nM] = excl + i * self.nBEAD[mol]
                    nM += 1
            else:
                for i in range(self.nMOL[mol]):
                    Excl_Mol[nM] = []
                    nM += 1
        self.Excl_Mol = Excl_Mol
        numBt = _np.sum(self.nBONDtype * self.nMOL)
        numAt = _np.sum(self.nANGLEtype * self.nMOL)
        numDt = _np.sum(self.nDIHEDRALtype * self.nMOL)
        allBondType = _np.zeros(numBt, dtype=int)
        allBondTypeIdx = _np.zeros(numBt, dtype=object)
        allAngleType = _np.zeros(numAt, dtype=int)
        allAngleTypeIdx = _np.zeros(numAt, dtype=object)
        allDihedralType = _np.zeros(numDt, dtype=int)
        allDihedralTypeIdx = _np.zeros(numDt, dtype=object)
        nBt = 0
        nAt = 0
        nDt = 0
        for mol in range(len(self.MOL)):
            for n in range(self.nMOL[mol]):
                allBondType[nBt: nBt + self.nBONDtype[mol]] = self.BONDtype[mol]
                allBondTypeIdx[nBt: nBt + self.nBONDtype[mol]] = BONDtypeIdx[mol] + n * self.nBEAD[mol]
                nBt += self.nBONDtype[mol]
                allAngleType[nAt: nAt + self.nANGLEtype[mol]] = self.ANGLEtype[mol]
                allAngleTypeIdx[nAt: nAt + self.nANGLEtype[mol]] = ANGLEtypeIdx[mol] + n * self.nBEAD[mol]
                nAt += self.nANGLEtype[mol]
                allDihedralType[nDt: nDt + self.nDIHEDRALtype[mol]] = self.DIHEDRALtype[mol]
                allDihedralTypeIdx[nDt: nDt + self.nDIHEDRALtype[mol]] = DIHEDRALtypeIdx[mol] + n * self.nBEAD[mol]
                nDt += self.nDIHEDRALtype[mol]
        self.BONDtypeSet = _np.unique(allBondType)
        self.BONDtypeSetIdx = _np.zeros(len(self.BONDtypeSet), dtype=object)
        self.BONDtypeSetName = _np.zeros([len(self.BONDtypeSet), 2], dtype='U5')
        BondTypeCat = _np.concatenate(self.BONDtype[:])
        BondTypeNameCat = _np.concatenate(self.BONDtypeName[:])
        for i in range(len(self.BONDtypeSet)):
            Ind = allBondType == self.BONDtypeSet[i]
            self.BONDtypeSetIdx[i] = _np.concatenate(allBondTypeIdx[Ind])
            for j in range(len(BondTypeCat)):
                if self.BONDtypeSet[i] == BondTypeCat[j]:
                    self.BONDtypeSetName[i] = BondTypeNameCat[j]
                    break
        self.ANGLEtypeSet = _np.unique(allAngleType)
        self.ANGLEtypeSetIdx = _np.zeros(len(self.ANGLEtypeSet), dtype=object)
        self.ANGLEtypeSetName = _np.zeros([len(self.ANGLEtypeSet), 3], dtype='U5')
        AngleTypeCat = _np.concatenate(self.ANGLEtype[:])
        AngleTypeNameCat = _np.concatenate(self.ANGLEtypeName[:])
        for i in range(len(self.ANGLEtypeSet)):
            Ind = allAngleType == self.ANGLEtypeSet[i]
            self.ANGLEtypeSetIdx[i] = _np.concatenate(allAngleTypeIdx[Ind])
            for j in range(len(AngleTypeCat)):
                if self.ANGLEtypeSet[i] == AngleTypeCat[j]:
                    self.ANGLEtypeSetName[i] = AngleTypeNameCat[j]
                    break
        self.DIHEDRALtypeSet = _np.unique(allDihedralType)
        self.DIHEDRALtypeSetIdx = _np.zeros(len(self.DIHEDRALtypeSet), dtype=object)
        self.DIHEDRALtypeSetName = _np.zeros([len(self.DIHEDRALtypeSet), 4], dtype='U5')
        DihedralTypeCat = _np.concatenate(self.DIHEDRALtype[:])
        DihedralTypeNameCat = _np.concatenate(self.DIHEDRALtypeName[:])
        for i in range(len(self.DIHEDRALtypeSet)):
            Ind = allDihedralType == self.DIHEDRALtypeSet[i]
            self.DIHEDRALtypeSetIdx[i] = _np.concatenate(allDihedralTypeIdx[Ind])
            for j in range(len(DihedralTypeCat)):
                if self.DIHEDRALtypeSet[i] == DihedralTypeCat[j]:
                    self.DIHEDRALtypeSetName[i] = DihedralTypeNameCat[j]
                    break
        ####################3
    def SetNonBondedIndex(self):
        BEADtype = _np.concatenate(self.BEADtype[:])
        self.BEADtypeSet = _np.zeros(1, dtype='<U5')
        for i in range(len(BEADtype)):
            if BEADtype[i] not in self.BEADtypeSet:
                self.BEADtypeSet = _np.append(self.BEADtypeSet, BEADtype[i])
        self.BEADtypeSet = self.BEADtypeSet[1:]
        self.BEADtypeSetIdx = _np.zeros(len(self.BEADtypeSet), dtype=object)
        allBEADtype = _np.zeros(_np.sum(self.nMOL * self.nBEAD), dtype='<U5')
        n = 0
        for mol in range(len(self.MOL)):
            for i in range(self.nMOL[mol]):
                allBEADtype[n: n + self.nBEAD[mol]] = self.BEADtype[mol]
                n += self.nBEAD[mol]
        for i in range(len(self.BEADtypeSet)):
            self.BEADtypeSetIdx[i] = _np.nonzero(allBEADtype == self.BEADtypeSet[i])[0]
        # NBit = _iter.combinations_with_replacement(self.BEADtypeSet, 2)
        # self.NonBONDED_Set = _np.array(list(NBit))
        nI = int(0.5 * len(self.BEADtypeSet) * (len(self.BEADtypeSet) + 1))
        self.NonBONDED_Set = _np.zeros([nI, 2], dtype='U5')
        self.NonBONDED_SetIdx = _np.zeros(nI, dtype=object)
        n = 0
        for i in range(len(self.BEADtypeSet)):
            for j in range(i, len(self.BEADtypeSet)):
                self.NonBONDED_Set[n, 0] = self.BEADtypeSet[i]
                self.NonBONDED_Set[n, 1] = self.BEADtypeSet[j]
                indI = self.BEADtypeSetIdx[i]
                indJ = self.BEADtypeSetIdx[j]
                pairs = _np.reshape(_np.array(_np.meshgrid(indI, indJ)).T, (-1, 2))
                exclPairs = _np.zeros(len(pairs), dtype=int)
                flag = 0
                for k in _np.arange(len(pairs)):
                    id1 = pairs[k, 0]
                    id2 = pairs[k, 1]
                    if id1 == id2:
                        exclPairs[flag] = k
                        flag += 1
                    elif self.MOLnum[id1] == self.MOLnum[id2]:
                        exclInd = self.Excl_Mol[self.MOLnum[id1]]
                        if len(exclInd) > 0:
                            if _np.any(_np.logical_and(exclInd[:, 0] == id1, exclInd[:, 1] == id2)):
                                exclPairs[flag] = k
                                flag += 1
                            elif _np.any(_np.logical_and(exclInd[:, 0] == id2, exclInd[:, 1] == id1)):
                                exclPairs[flag] = k
                                flag += 1
                self.NonBONDED_SetIdx[n] = _np.delete(pairs, exclPairs[0:flag], axis=0)
                n += 1

    ###################################3

    def calcBondsMap(self):
        self.BondsMap = _np.zeros(_np.sum(self.nMOL * self.nBEAD), dtype=object)
        BeadCumSum = _np.append([0], _np.cumsum(self.nBEAD * self.nMOL))
        n = 0
        for mol in range(len(self.MOL)):
            if self.nBONDtype[mol] > 0:
                bonds = _np.concatenate(self.BONDtypeIdx[mol])
                # bondsMap = _np.zeros(self.nBEAD[mol], dtype=object)
                bondsMap = _np.empty(self.nBEAD[mol], dtype=object)
                for i in range(self.nBEAD[mol]):
                    C0 = bonds[:, 0]
                    C1 = bonds[:, 1]
                    temp = _np.append(C0[C1 == i], C1[C0 == i])
                    bondsMap[i] = temp[temp > i] + BeadCumSum[mol]
                for j in range(self.nMOL[mol]):
                    self.BondsMap[n: n + self.nBEAD[mol]] = bondsMap + j * self.nBEAD[mol]
                    n += self.nBEAD[mol]
            # else:
            #     for j in range(self.nMOL[mol] * self.nBEAD[mol]):
            #         self.BondsMap[n] = []
            #         n += 1
#####################################################
#####################################################
class _cTAB():
    def __init__(self, inFile, mode):
        inFile = str(inFile)
        self._mode = str(mode)
        if inFile.endswith('.hist'):
            self.Type = 'HIST'
        elif inFile.endswith('.dist'):
            self.Type = 'DIST'
        elif inFile.endswith('.pot'):
            self.Type = 'POT'
        elif inFile.endswith('.dpot'):
            self.Type = 'dPOT'
        else:
            raise _scg4pyError('input file should be one of: ".hist", ".dist", ".pot", ".dpot"')
        if self._mode == 'r':
            fid = open(inFile, mode='r')
            tabL = fid.readlines()
            fid.close()
            nNBtab = nBtab = nAtab = nDtab = 0
            for line in tabL:
                lsp = line.split('#')[0]
                if 'NAME' in lsp:
                    if lsp.split('=')[1].split('_')[0].strip() == 'Non-Bonded':
                        nNBtab += 1
                    elif lsp.split('=')[1].split('_')[0].strip() == 'Bond':
                        nBtab += 1
                    elif lsp.split('=')[1].split('_')[0].strip() == 'Angle':
                        nAtab += 1
                    elif lsp.split('=')[1].split('_')[0].strip() == 'Dihedral':
                        nDtab += 1
            self.BondName = _np.zeros([nBtab, 2], dtype='U5')
            self.BondType = _np.zeros(nBtab, dtype=int)
            self.BondX = _np.zeros(nBtab, dtype=object)
            self.BondY = _np.zeros(nBtab, dtype=object)
            self.AngleName = _np.zeros([nAtab, 3], dtype='U5')
            self.AngleType = _np.zeros(nAtab, dtype=int)
            self.AngleX = _np.zeros(nAtab, dtype=object)
            self.AngleY = _np.zeros(nAtab, dtype=object)
            self.DihedralName = _np.zeros([nDtab, 4], dtype='U5')
            self.DihedralType = _np.zeros(nDtab, dtype=int)
            self.DihedralX = _np.zeros(nDtab, dtype=object)
            self.DihedralY = _np.zeros(nDtab, dtype=object)
            self.NonBondType = _np.zeros([nNBtab, 2], dtype='U5')
            self.NonBondX = _np.zeros(nNBtab, dtype=object)
            self.NonBondY = _np.zeros(nNBtab, dtype=object)
            line = 0
            for i in range(nNBtab):
                while 'Non-Bonded_' not in tabL[line].split('#')[0]:
                    line += 1
                line += 1
                self.NonBondType[i, 0] = tabL[line].split('#')[0].split('=')[1].split(',')[0].strip()
                self.NonBondType[i, 1] = tabL[line].split('#')[0].split('=')[1].split(',')[1].strip()
                line += 1
                self.NonBondX[i] = _np.array(tabL[line].split(), dtype=float)
                line += 1
                self.NonBondY[i] = _np.array(tabL[line].split(), dtype=float)
            line = 0
            for i in range(nBtab):
                while 'Bond_' not in tabL[line].split('#')[0]:
                    line += 1
                lsp = tabL[line].split('#')[0].split('=')[1].strip()
                self.BondName[i, 0] = lsp.split('Bond_')[1].split('-')[0].strip()
                self.BondName[i, 1] = lsp.split('Bond_')[1].split('-')[1].strip()
                line += 1
                self.BondType[i] = int(tabL[line].split('#')[0].split('=')[1].strip())
                line += 1
                self.BondX[i] = _np.array(tabL[line].split(), dtype=float)
                line += 1
                self.BondY[i] = _np.array(tabL[line].split(), dtype=float)
            line = 0
            for i in range(nAtab):
                while 'Angle_' not in tabL[line].split('#')[0]:
                    line += 1
                lsp = tabL[line].split('#')[0].split('=')[1].strip()
                self.AngleName[i, 0] = lsp.split('Angle_')[1].split('-')[0].strip()
                self.AngleName[i, 1] = lsp.split('Angle_')[1].split('-')[1].strip()
                self.AngleName[i, 2] = lsp.split('Angle_')[1].split('-')[2].strip()
                line += 1
                self.AngleType[i] = int(tabL[line].split('#')[0].split('=')[1].strip())
                line += 1
                self.AngleX[i] = _np.array(tabL[line].split(), dtype=float)
                line += 1
                self.AngleY[i] = _np.array(tabL[line].split(), dtype=float)
            line = 0
            for i in range(nDtab):
                while 'Dihedral_' not in tabL[line].split('#')[0]:
                    line += 1
                lsp = tabL[line].split('#')[0].split('=')[1].strip()
                self.DihedralName[i, 0] = lsp.split('Dihedral_')[1].split('-')[0].strip()
                self.DihedralName[i, 1] = lsp.split('Dihedral_')[1].split('-')[1].strip()
                self.DihedralName[i, 2] = lsp.split('Dihedral_')[1].split('-')[2].strip()
                self.DihedralName[i, 3] = lsp.split('Dihedral_')[1].split('-')[3].strip()
                line += 1
                self.DihedralType[i] = int(tabL[line].split('#')[0].split('=')[1].strip())
                line += 1
                self.DihedralX[i] = _np.array(tabL[line].split(), dtype=float)
                line += 1
                self.DihedralY[i] = _np.array(tabL[line].split(), dtype=float)
        elif self._mode == 'w':
            self._fileName = inFile
        else:
            raise _scg4pyError('opening mode should be one of "r" or "w".')

    def setattr(self, NonBondType, NonBondX, NonBondY, BondName=None, BondType=None, BondX=None, BondY=None,
                AngleName=None, AngleType=None, AngleX=None, AngleY=None,
                DihedralName=None, DihedralType=None, DihedralX=None, DihedralY=None):
        if self._mode == 'w':
            self.BondName = BondName
            self.BondType = BondType
            self.BondX = BondX
            self.BondY = BondY
            self.AngleName = AngleName
            self.AngleType = AngleType
            self.AngleX = AngleX
            self.AngleY = AngleY
            self.DihedralName = DihedralName
            self.DihedralType = DihedralType
            self.DihedralX = DihedralX
            self.DihedralY = DihedralY
            self.NonBondType = NonBondType
            self.NonBondX = NonBondX
            self.NonBondY = NonBondY

    def trim(self, x, y, yMin, BothSide):
        I = 0
        for i in range(len(y)):
            if y[i] > yMin:
                I = i
                break
        J = len(y) - 1
        if BothSide:
            for j in range(len(y) - 1, 0, -1):
                if y[j] > yMin:
                    J = j
                    break
            if J == len(y) - 1:
                X = x[I:]
                Y = y[I:]
            else:
                J += 1
                X = x[I:J]
                Y = y[I:J]
        else:
            X = x[I:]
            Y = y[I:]
        return X, Y

    def smooth(self, x, y, sig):
        if sig > 0:
            SIG = sig * _np.mean(_np.diff(x))
            yy = _np.zeros(_np.size(x), dtype=float)
            for i in _np.arange(_np.size(yy)):
                expX = _np.exp((-1 * (x[i] - x) ** 2) / (2 * SIG ** 2))
                Z = _np.sum(expX)
                yy[i] = _np.sum(y * expX) / Z
        else:
            yy = y
        return yy

    def write(self):
        if self._mode == 'w':
            self._fid = open(self._fileName, mode='w')
            for i in range(len(self.NonBondType)):
                t1 = self.NonBondType[i, 0]
                t2 = self.NonBondType[i, 1]
                if len(self.NonBondX[i]) != len(self.NonBondY[i]):
                    mess = 'length of "x" and "y" are not equal in {}'.format('Non-Bonded_' + t1 + '-' + t2)
                    raise _scg4pyError(mess)
                self._fid.write('NAME = {}\n'.format('Non-Bonded_' + t1 + '-' + t2))
                self._fid.write('TYPE = {} , {}\n'.format(t1, t2))
                for j in range(len(self.NonBondX[i])):
                    self._fid.write('{:.12e} '.format(self.NonBondX[i][j]))
                self._fid.write('\n')
                for j in range(len(self.NonBondY[i])):
                    self._fid.write('{:.12e} '.format(self.NonBondY[i][j]))
                self._fid.write('\n\n')
            if self.BondType is not None:
                for i in range(len(self.BondType)):
                    if i == 0:
                        self._fid.write('###################\n###################\n\n')
                    n1 = self.BondName[i, 0]
                    n2 = self.BondName[i, 1]
                    if len(self.BondX[i]) != len(self.BondY[i]):
                        mess = 'length of "x" and "y" are not equal in {}'.format('Bond_' + n1 + '-' + n2)
                        raise _scg4pyError(mess)
                    self._fid.write('NAME = {}\n'.format('Bond_' + n1 + '-' + n2))
                    self._fid.write('TYPE = {}\n'.format(self.BondType[i]))
                    for j in range(len(self.BondX[i])):
                        self._fid.write('{:.12e} '.format(self.BondX[i][j]))
                    self._fid.write('\n')
                    for j in range(len(self.BondY[i])):
                        self._fid.write('{:.12e} '.format(self.BondY[i][j]))
                    self._fid.write('\n\n')
            if self.AngleType is not None:
                for i in range(len(self.AngleType)):
                    if i == 0:
                        self._fid.write('###################\n###################\n\n')
                    n1 = self.AngleName[i, 0]
                    n2 = self.AngleName[i, 1]
                    n3 = self.AngleName[i, 2]
                    if len(self.AngleX[i]) != len(self.AngleY[i]):
                        mess = 'length of "x" and "y" are not equal in {}'.format('Angle_' + n1 + '-' + n2 + '-' + n3)
                        raise _scg4pyError(mess)
                    self._fid.write('NAME = {}\n'.format('Angle_' + n1 + '-' + n2 + '-' + n3))
                    self._fid.write('TYPE = {}\n'.format(self.AngleType[i]))
                    for j in range(len(self.AngleX[i])):
                        self._fid.write('{:.12e} '.format(self.AngleX[i][j]))
                    self._fid.write('\n')
                    for j in range(len(self.AngleY[i])):
                        self._fid.write('{:.12e} '.format(self.AngleY[i][j]))
                    self._fid.write('\n\n')
            if self.DihedralType is not None:
                for i in range(len(self.DihedralType)):
                    if i == 0:
                        self._fid.write('###################\n###################\n\n')
                    n1 = self.DihedralName[i, 0]
                    n2 = self.DihedralName[i, 1]
                    n3 = self.DihedralName[i, 2]
                    n4 = self.DihedralName[i, 3]
                    if len(self.DihedralX[i]) != len(self.DihedralY[i]):
                        mess = 'length of "x" and "y" are not equal in {}'.format(
                            'Dihedral_' + n1 + '-' + n2 + '-' + n3 + '-' + n4)
                        raise _scg4pyError(mess)
                    self._fid.write('NAME = {}\n'.format(
                        'Dihedral_' + n1 + '-' + n2 + '-' + n3 + '-' + n4))
                    self._fid.write('TYPE = {}\n'.format(self.DihedralType[i]))
                    for j in range(len(self.DihedralX[i])):
                        self._fid.write('{:.12e} '.format(self.DihedralX[i][j]))
                    self._fid.write('\n')
                    for j in range(len(self.DihedralY[i])):
                        self._fid.write('{:.12e} '.format(self.DihedralY[i][j]))
                    self._fid.write('\n\n')
            self._fid.close()

    def extrpPot_Bond(self, i, binW, maxF, cutoff):
        if self.Type == 'POT':
            pass
        else:
            mess = 'only ".pot" files can be extrapolated'
            raise _scg4pyError(mess)
        x = self.BondX[i]
        y = self.BondY[i]
        npoint = int(round(cutoff / binW)) + 1
        X = _np.linspace(0.0, cutoff, npoint)
        dX = _np.mean(_np.diff(X))
        X = _np.concatenate([[-dX], X, [cutoff + dX]])
        # 1- x & pot extension
        dx = _np.mean(_np.diff(x))
        xl = _np.array([x[0] - 3 * dx, x[0] - 2 * dx, x[0] - dx])
        yl = ((y[1] - y[0]) / dx) * (xl - x[0]) + y[0]
        xr = _np.array([x[-1] + dx, x[-1] + 2 * dx, x[-1] + 3 * dx])
        yr = ((y[-1] - y[-2]) / dx) * (xr - x[-1]) + y[-1]
        x_ext = _np.concatenate([xl, x, xr])
        y_ext = _np.concatenate([yl, y, yr])
        # 2- pot interpolation
        xC = X[_np.logical_and(X > x[0], X < x[-1])]
        # xC = X[(X > x[0]) * (X < x[-1])]
        Rbf = _spint.Rbf(x_ext, y_ext)
        yC = Rbf(xC)
        fC = (yC[2:] - yC[0:-2]) / (2 * dX)
        xC = xC[1:-1]
        yC = yC[1:-1]
        # 3- pot extrapolation (LHS)
        xL = X[X < xC[0]]
        m1 = -1 * maxF
        x1 = 0.0
        x2 = xC[0]
        m2 = fC[0]
        y2 = yC[0]
        mat = _np.array([[2 * x1, 1, 0], [2 * x2, 1, 0], [x2 ** 2, x2, 1]], dtype=float)
        vec = _np.array([m1, m2, y2])
        f = _np.dot(_np.linalg.pinv(mat), vec)
        a, b, c = f[0], f[1], f[2]
        yL = a * xL ** 2 + b * xL + c
        # 4- pot extrapolation (RHS)
        xR = X[X > xC[-1]]
        x1 = xC[-1]
        y1 = yC[-1]
        m1 = fC[-1]
        b = m1 - 2 * a * x1
        c = y1 - a * x1 ** 2 - b * x1
        yR = a * xR ** 2 + b * xR + c
        # 5- Pot calculation
        Pot = _np.concatenate([yL, yC, yR])
        # 6- Force calculation
        Force = -1 * (Pot[2:] - Pot[0:-2]) / (2 * dX)
        X = X[1:-1]
        Pot = Pot[1:-1]
        return X, Pot, Force

    def extrpPot_Angle(self, i, binW, maxF):
        if self.Type == 'POT':
            pass
        else:
            mess = 'only ".pot" files can be extrapolated'
            raise _scg4pyError(mess)
        x = self.AngleX[i]
        y = self.AngleY[i]
        npoint = int(round(180 / binW)) + 1
        X = _np.linspace(0.0, 180, npoint)
        dX = _np.mean(_np.diff(X))
        X = _np.concatenate([[-dX], X, [180 + dX]])
        # 1- x & pot extension
        dx = _np.mean(_np.diff(x))
        xl = _np.array([x[0] - 3 * dx, x[0] - 2 * dx, x[0] - dx])
        yl = ((y[1] - y[0]) / dx) * (xl - x[0]) + y[0]
        xr = _np.array([x[-1] + dx, x[-1] + 2 * dx, x[-1] + 3 * dx])
        yr = ((y[-1] - y[-2]) / dx) * (xr - x[-1]) + y[-1]
        x_ext = _np.concatenate([xl, x, xr])
        y_ext = _np.concatenate([yl, y, yr])
        # 2- pot interpolation
        xC = X[_np.logical_and(X > x[0], X < x[-1])]
        # xC = X[(X > x[0]) * (X < x[-1])]
        Rbf = _spint.Rbf(x_ext, y_ext)
        yC = Rbf(xC)
        fC = (yC[2:] - yC[0:-2]) / (2 * dX)
        xC = xC[1:-1]
        yC = yC[1:-1]
        # 3- pot extrapolation (LHS)
        xL = X[X < xC[0]]
        m1 = -1 * maxF
        x1 = 0.0
        x2 = xC[0]
        m2 = fC[0]
        y2 = yC[0]
        mat = _np.array([[2 * x1, 1, 0], [2 * x2, 1, 0], [x2 ** 2, x2, 1]], dtype=float)
        vec = _np.array([m1, m2, y2])
        f = _np.dot(_np.linalg.pinv(mat), vec)
        a, b, c = f[0], f[1], f[2]
        yL = a * xL ** 2 + b * xL + c
        # 4- pot extrapolation (RHS)
        m2 = abs(m1)
        if len(self.DihedralType) > 0:
            yC = yC[xC < 175]
            xC = xC[xC < 175]
        else:
            if xC[-1] > 170:
                m2 = 0.0
        xR = X[X > xC[-1]]
        x2 = 180.0
        m1 = fC[-1]
        y1 = yC[-1]
        x1 = xC[-1]
        mat = _np.array([[x1 ** 2, x1, 1], [2 * x1, 1, 0], [2 * x2, 1, 0]], dtype=float)
        vec = _np.array([y1, m1, m2])
        f = _np.dot(_np.linalg.pinv(mat), vec)
        a, b, c = f[0], f[1], f[2]
        yR = a * xR ** 2 + b * xR + c
        # 4- Pot calculation
        Pot = _np.concatenate([yL, yC, yR])
        # 5- Force calculation
        Force = -1 * (Pot[2:] - Pot[0:-2]) / (2 * dX)
        X = X[1:-1]
        Pot = Pot[1:-1]
        return X, Pot, Force

    def extrpPot_Dihedral(self, i, binW):
        x = self.DihedralX[i]
        y = self.DihedralY[i]
        npoint = int(round(360 / binW)) + 1
        X = _np.linspace(-180, 180, npoint)
        dX = _np.mean(_np.diff(X))
        # 1- x & pot extension
        x_ext = _np.concatenate([x - 2 * 180, x, x + 2 * 180])
        y_ext = _np.concatenate([y, y, y])
        # 2- Pot interpolation
        Rbf = _spint.Rbf(x_ext, y_ext)
        Pot = Rbf(X)
        # 3- Force calculation
        Force = _np.zeros(len(X))
        Force[1:-1] = -1 * (Pot[2:] - Pot[0:-2]) / (2 * dX)
        Force[0] = -1 * (Pot[1] - Pot[-2]) / (2 * dX)
        Force[-1] = Force[0]
        return X, Pot, Force

    def extrpPot_NonBonded(self, i, binW, maxF, cutoff, skin):
        # 1- x & pot extension
        x = self.NonBondX[i]
        y = self.NonBondY[i]
        npoint = int(round(cutoff / binW)) + 1
        X = _np.linspace(0.0, cutoff, npoint)
        dX = _np.mean(_np.diff(X))
        X = _np.concatenate([[-dX], X, [cutoff + dX]])
        dx = _np.mean(_np.diff(x))
        xl = _np.array([x[0] - 3 * dx, x[0] - 2 * dx, x[0] - dx])
        yl = ((y[1] - y[0]) / dx) * (xl - x[0]) + y[0]
        xr = _np.array([x[-1] + dx, x[-1] + 2 * dx, x[-1] + 3 * dx])
        yr = ((y[-1] - y[-2]) / dx) * (xr - x[-1]) + y[-1]
        x_ext = _np.concatenate([xl, x, xr])
        y_ext = _np.concatenate([yl, y, yr])
        # 2- pot interpolation
        xC = X[(X > x[0]) * (X < x[-1])]
        Rbf = _spint.Rbf(x_ext, y_ext)
        yC = Rbf(xC)
        fC = (yC[2:] - yC[0:-2]) / (2 * dX)
        xC = xC[1:-1]
        yC = yC[1:-1]
        # 3- xx & yy triming --> xx <= (cutoff-skin)
        yC = yC[xC <= (cutoff - skin)]
        xC = xC[xC <= (cutoff - skin)]
        # 4- pot extrapolation (LHS)
        xL = X[X < xC[0]]
        m1 = -1 * maxF
        x1 = 0.0
        x2 = xC[0]
        m2 = fC[0]
        y2 = yC[0]
        mat = _np.array([[2 * x1, 1, 0], [2 * x2, 1, 0], [x2 ** 2, x2, 1]], dtype=float)
        vec = _np.array([m1, m2, y2])
        f = _np.dot(_np.linalg.pinv(mat), vec)
        a, b, c = f[0], f[1], f[2]
        yL = a * xL ** 2 + b * xL + c
        # 5- pot extrapolation (RHS)
        xR = X[X > xC[-1]]
        x1 = xC[-1]
        x2 = cutoff
        y1 = yC[-1]
        y2 = 0
        m1 = fC[-1]
        m2 = 0
        mat = _np.array([[x1 ** 5, x1 ** 3, x1, 1], [x2 ** 5, x2 ** 3, x2, 1],
                         [5 * x1 ** 4, 3 * x1 ** 2, 1, 0], [5 * x2 ** 4, 3 * x2 ** 2, 1, 0]], dtype=float)
        vec = _np.array([y1, y2, m1, m2])
        f = _np.dot(_np.linalg.pinv(mat), vec)
        a, b, c, d = f[0], f[1], f[2], f[3]
        yR = a * xR ** 5 + b * xR ** 3 + c * xR + d
        yR[-2:] = 0.0
        # 6- Pot calculation
        Pot = _np.concatenate([yL, yC, yR])
        # 7- Force calculation
        Force = -1 * (Pot[2:] - Pot[0:-2]) / (2 * dX)
        Force[-1] = 0.0
        X = X[1:-1]
        Pot = Pot[1:-1]
        return X, Pot, Force
#####################################################
#####################################################
class _cDFs():
    def __init__(self):
        pass

    def calcNB(self, xs, ys, zs, setIdx, boxMat, unwrapped=False):
        if len(boxMat) == 15:
            lx = boxMat['xhi'] - boxMat['xlo']
            ly = boxMat['yhi'] - boxMat['ylo']
            lz = boxMat['zhi'] - boxMat['zlo']
            xy, xz, yz = boxMat['xy'], boxMat['xz'], boxMat['yz']
        else:
            lx, ly, lz, xy, xz, yz = boxMat[0], boxMat[1], boxMat[2],\
                                     boxMat[3], boxMat[4], boxMat[5]
        if unwrapped:
            xlo, ylo, zlo = boxMat['xlo'], boxMat['ylo'], boxMat['zlo']
            x, y, z = xs, ys, zs
            xs = (x - xlo) / lx - (y - ylo) * xy / (lx * ly) + (z - zlo) * (yz * xy - xz * ly) / (lx * ly * lz)
            ys = (y - ylo) / ly - yz * (z - zlo) / (ly * lz)
            zs = (z - zlo) / lz
        pairData = _np.zeros(len(setIdx), dtype=object)
        for i in range(len(setIdx)):
            indA = setIdx[i][:, 0]
            indB = setIdx[i][:, 1]
            xA, yA, zA = xs[indA], ys[indA], zs[indA]
            xB, yB, zB = xs[indB], ys[indB], zs[indB]
            dx = xA - xB - _np.rint(xA - xB)
            dy = yA - yB - _np.rint(yA - yB)
            dz = zA - zB - _np.rint(zA - zB)
            dX = dx * lx + dy * xy + dz * xz
            dY = dy * ly + dz * yz
            dZ = dz * lz
            pairData[i] = _np.sqrt(dX ** 2 + dY ** 2 + dZ ** 2)
        return pairData

    def calcB(self, x, y, z, setIdx):
        bondData = _np.zeros(len(setIdx), dtype=object)
        for i in range(len(setIdx)):
            indA = setIdx[i][:, 0]
            indB = setIdx[i][:, 1]
            xA, yA, zA = x[indA], y[indA], z[indA]
            xB, yB, zB = x[indB], y[indB], z[indB]
            dx = xA - xB
            dy = yA - yB
            dz = zA - zB
            bondData[i] = _np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return bondData

    def calcA(self, x, y, z, setIdx):
        AngleData = _np.zeros(len(setIdx), dtype=object)
        for i in range(len(setIdx)):
            indA = setIdx[i][:, 0]
            indB = setIdx[i][:, 1]
            indC = setIdx[i][:, 2]
            xA, yA, zA = x[indA], y[indA], z[indA]
            xB, yB, zB = x[indB], y[indB], z[indB]
            xC, yC, zC = x[indC], y[indC], z[indC]
            xBA, yBA, zBA = xA - xB, yA - yB, zA - zB
            xBC, yBC, zBC = xC - xB, yC - yB, zC - zB
            cosTheta = (xBA * xBC + yBA * yBC + zBA * zBC)
            cosTheta = cosTheta / (_np.sqrt(xBA ** 2 + yBA ** 2 + zBA ** 2) * _np.sqrt(xBC ** 2 + yBC ** 2 + zBC ** 2))
            cosTheta[cosTheta > 1.0] = 1.0
            cosTheta[cosTheta < -1.0] = -1.0
            AngleData[i] = _np.rad2deg(_np.arccos(cosTheta))
        return AngleData

    def calcD(self, x, y, z, setIdx):
        DihedralData = _np.zeros(len(setIdx), dtype=object)
        for i in range(len(setIdx)):
            indA = setIdx[i][:, 0]
            indB = setIdx[i][:, 1]
            indC = setIdx[i][:, 2]
            indD = setIdx[i][:, 3]
            xA, yA, zA = x[indA], y[indA], z[indA]
            xB, yB, zB = x[indB], y[indB], z[indB]
            xC, yC, zC = x[indC], y[indC], z[indC]
            xD, yD, zD = x[indD], y[indD], z[indD]
            xAB, yAB, zAB = xB - xA, yB - yA, zB - zA
            xBC, yBC, zBC = xC - xB, yC - yB, zC - zB
            xCD, yCD, zCD = xD - xC, yD - yC, zD - zC
            xABxBC = yAB * zBC - zAB * yBC
            yABxBC = zAB * xBC - xAB * zBC
            zABxBC = xAB * yBC - yAB * xBC
            xBCxCD = yBC * zCD - zBC * yCD
            yBCxCD = zBC * xCD - xBC * zCD
            zBCxCD = xBC * yCD - yBC * xCD
            abs_ABxBC = _np.sqrt(xABxBC ** 2 + yABxBC ** 2 + zABxBC ** 2)
            xn1, yn1, zn1 = xABxBC / abs_ABxBC, yABxBC / abs_ABxBC, zABxBC / abs_ABxBC
            abs_BCxCD = _np.sqrt(xBCxCD ** 2 + yBCxCD ** 2 + zBCxCD ** 2)
            xn2, yn2, zn2 = xBCxCD / abs_BCxCD, yBCxCD / abs_BCxCD, zBCxCD / abs_BCxCD
            abs_BC = _np.sqrt(xBC ** 2 + yBC ** 2 + zBC ** 2)
            xBCn, yBCn, zBCn = xBC / abs_BC, yBC / abs_BC, zBC / abs_BC
            xu = yBCn * zn2 - zBCn * yn2
            yu = zBCn * xn2 - xBCn * zn2
            zu = xBCn * yn2 - yBCn * xn2
            n1_dot_u = xn1 * xu + yn1 * yu + zn1 * zu
            n1_dor_n2 = xn1 * xn2 + yn1 * yn2 + zn1 * zn2
            phi = -1 * _np.arctan2(n1_dot_u, n1_dor_n2)
            DihedralData[i] = _np.rad2deg(phi)
        return DihedralData

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

def Mapping(inFile, outTop=None, readTRJ=True, coulomb=False):
    '''This program reads an input mapping file, then translates the atomistic trajectory to the coarse-grained. the coordinates of each CG site are calculated by the center of mass of involved atoms. Also, a CG topology file and CG structures of each molecule type are generated as output. This module supports a range of trajectory file format including ’lammpstrj’, ’pdb’, ’gro’, ’xtc’, and ’dcd’.

Parameters:
• inFile: Input mapping file (*.map).
• outTop: Output CG topology file. If None, it is specified as the name of the input mapping file.
• readTRJ: If False, generates only CG topology file.
• coulomb: If True, the electric charge of each bead is calculated from the total electric charge of the atoms involved in that bead otherwise, it is considered zero.

Outputs:
• The CG topology file ([outTop].CGtop).
• The CG trajectory file.
• The log file ([inFile].log) .
• The last snapshot of the CG trajectory (*.pbd).
• The CG structure of each molecule type in different files (*.pdb).'''

    ##########################################
    ##########################################

    inFile = _Rstrip(inFile, '.map')
    if outTop == None:
        outTop = inFile + '.CGtop'
    else:
        outTop = _Rstrip(outTop, '.CGtop') + '.CGtop'
    inFile += '.map'
    sysMap = _cMAP(inFile)
    sysMap.do_classification()

    AtomCumSum = _np.append([0], _np.cumsum(sysMap.nATOM * sysMap.nMOL))
    BeadCumSum = _np.append([0], _np.cumsum(sysMap.nBEAD * sysMap.nMOL))
    allATOMmass = _np.zeros(_np.sum(sysMap.nMOL * sysMap.nATOM), dtype=float)
    newBEADmap = _np.zeros(len(sysMap.MOL), dtype=object)
    for mol in range(len(sysMap.MOL)):
        newBEADmap[mol] = sysMap.BEADmap[mol] + AtomCumSum[mol]
    nAllBead = int(_np.sum(sysMap.nMOL * sysMap.nBEAD))
    allBEADmap = _np.zeros(nAllBead, dtype=object)
    allMOLid = _np.zeros(nAllBead, dtype=int)
    allMOLname = _np.zeros(nAllBead, dtype='<U5')
    allBEADname = _np.zeros(nAllBead, dtype='<U5')
    allLMPtypeId = _np.zeros(nAllBead, dtype=int)
    allBEADid = _np.arange(1, nAllBead + 1)
    sysName = '-'.join(sysMap.MOL)
    nAtom = _np.sum(sysMap.nMOL * sysMap.nATOM)
    counter = 1
    m = 0
    for mol in range(len(sysMap.MOL)):
        mass = sysMap.ATOMms[mol]
        allATOMmass[AtomCumSum[mol]:AtomCumSum[mol + 1]] = _np.tile(mass, sysMap.nMOL[mol])
        allMOLname[BeadCumSum[mol]:BeadCumSum[mol + 1]] = _np.tile(sysMap.MOL[mol], sysMap.nBEAD[mol] * sysMap.nMOL[mol])
        allBEADname[BeadCumSum[mol]:BeadCumSum[mol + 1]] = _np.tile(sysMap.BEADname[mol], sysMap.nMOL[mol])
        allLMPtypeId[BeadCumSum[mol]:BeadCumSum[mol + 1]] = _np.tile(sysMap.LMPtype[mol], sysMap.nMOL[mol])
        for n in range(sysMap.nMOL[mol]):
            allBEADmap[m: m + sysMap.nBEAD[mol]] = newBEADmap[mol] + n * sysMap.nATOM[mol]
            allMOLid[m: m + sysMap.nBEAD[mol]] = counter * _np.ones(sysMap.nBEAD[mol], dtype=int)
            m += sysMap.nBEAD[mol]
            counter += 1

    if readTRJ:
        if sysMap.AATRAJ == sysMap.CGTRAJ:
            mess = 'input and output trajectory must be different.'
            raise _scg4pyError(mess)
        AAtraj = _cTRAJ(sysMap.AATRAJ, 'r')
        AAtraj.read(nAtom=nAtom)
        CGtraj = _cTRAJ(sysMap.CGTRAJ, 'w')
        outPath = _os.path.split(sysMap.CGTRAJ)[0]
        if not outPath:
            outPath = './'
        else:
            outPath = outPath + '/'
        print('')
        n = 0
        boxMatAv = _np.zeros(6, dtype=float)
        print('output CG trajectory: {}'.format(sysMap.CGTRAJ))
        while True:
            if n % 2 == 0:
                if AAtraj.time:
                    print('\rTime = {}'.format(AAtraj.time), end='')
                elif AAtraj.timeStep:
                    print('\rTimeStep = {}'.format(AAtraj.timeStep), end='')
            if n == 0:
                print()
            n += 1
            boxMatAv = boxMatAv + AAtraj.boxMat
            xCG = _np.zeros(nAllBead)
            yCG = _np.zeros(nAllBead)
            zCG = _np.zeros(nAllBead)
            if AAtraj.Type == 'LMP':
                if AAtraj.xu is not None:
                    xAA = AAtraj.xu
                    yAA = AAtraj.yu
                    zAA = AAtraj.zu
                elif (AAtraj.xs is not None) and (AAtraj.ix is not None):
                    AAtraj.unwrapLMPsnap()
                    xAA = AAtraj.xu
                    yAA = AAtraj.yu
                    zAA = AAtraj.zu
                elif AAtraj.x is not None:
                    xAA = AAtraj.x
                    yAA = AAtraj.y
                    zAA = AAtraj.z
                elif AAtraj.xs is not None:
                    AAtraj.unscaleSnap()
                    xAA = AAtraj.x
                    yAA = AAtraj.y
                    zAA = AAtraj.z
                else:
                    raise _scg4pyError('Error in reading LAMMPS trajectory.')
            else:
                xAA = AAtraj.x
                yAA = AAtraj.y
                zAA = AAtraj.z

            for i in _np.arange(len(allBEADmap)):
                beadMass = _np.sum(allATOMmass[allBEADmap[i]])
                xCG[i] = _np.sum(allATOMmass[allBEADmap[i]] * xAA[allBEADmap[i]]) / beadMass
                yCG[i] = _np.sum(allATOMmass[allBEADmap[i]] * yAA[allBEADmap[i]]) / beadMass
                zCG[i] = _np.sum(allATOMmass[allBEADmap[i]] * zAA[allBEADmap[i]]) / beadMass
            if AAtraj.time is not None:
                CGtraj.time = AAtraj.time
            else:
                CGtraj.time = n - 1
            if AAtraj.timeStep is not None:
                CGtraj.timeStep = AAtraj.timeStep
            else:
                CGtraj.timeStep = n - 1
            CGtraj.nAtom = len(xCG)
            CGtraj.resId = allMOLid
            CGtraj.resName = allMOLname
            CGtraj.atId = allBEADid
            CGtraj.atName = allBEADname
            CGtraj.atType = allLMPtypeId
            CGtraj.boxMat = AAtraj.boxMat
            CGtraj.x = xCG
            CGtraj.y = yCG
            CGtraj.z = zCG
            CGtraj.write(modelN=n, sysName=sysName)
            AAtraj.read(nAtom=nAtom)
            if AAtraj.eof:
                AAtraj.close()
                CGtraj.close()
                break
        if AAtraj.time:
            print('\rTime = {}'.format(AAtraj.time))
        elif AAtraj.timeStep:
            print('\rTimeStep = {}'.format(AAtraj.timeStep))
        print('\n\nnumber of snapshots : {}\noutput files:'.format(n))

        boxMatAv = boxMatAv / n
        lx, ly, lz, xy, xz, yz = boxMatAv[0], boxMatAv[1], boxMatAv[2], boxMatAv[3], boxMatAv[4], boxMatAv[5]
        A = lx
        B = _np.sqrt(ly ** 2 + xy ** 2)
        C = _np.sqrt(lz ** 2 + xz ** 2 + yz ** 2)
        alpha = _np.rad2deg(_np.arccos((xy * xz + ly * yz) / (B * C)))
        beta = _np.rad2deg(_np.arccos(xz / C))
        gamma = _np.rad2deg(_np.arccos(xy / B))
        boxABC = _np.array([A, B, C, alpha, beta, gamma])
        vol = lx * ly * lz
        logName = outTop[0:-6] + '.log'
        with open(logName, 'w') as mapLog:
            print('   log file: {}'.format(logName))
            mapLog.write('Number of snapshots : {}\n'.format(n))
            mapLog.write('Box length [Angstrom] : \n')
            mapLog.write('\t[Lx, Ly, Lz, XY, XZ, YZ] :\n')
            mapLog.write('\t\t{}\n'.format(boxMatAv))
            mapLog.write('\t[A, B, C, alpha, beta, gamma] :\n')
            mapLog.write('\t\t{}\n'.format(boxABC))
            mapLog.write('Box volume [Angstrom^3] : \n\t{}\n'.format(vol))
        lastPdbName = outPath + 'CGlastSnap_' + sysName + '.pdb'
        lastSnap = _cTRAJ(lastPdbName, 'w')
        print('   last snapshot: {}'.format(lastPdbName))
        lastSnap.time = CGtraj.time
        lastSnap.nAtom = CGtraj.nAtom
        lastSnap.resId = CGtraj.resId
        lastSnap.resName = CGtraj.resName
        lastSnap.atId = CGtraj.atId
        lastSnap.atName = CGtraj.atName
        lastSnap.boxMat = boxMatAv
        lastSnap.x = CGtraj.x
        lastSnap.y = CGtraj.y
        lastSnap.z = CGtraj.z
        lastSnap.write(modelN=1, sysName=sysName)
        lastSnap.close()
        ind = _np.append(0, _np.cumsum(sysMap.nBEAD * sysMap.nMOL)[0:-1])
        for mol in range(len(sysMap.MOL)):
            x = xCG[ind[mol]:ind[mol] + sysMap.nBEAD[mol]]
            y = yCG[ind[mol]:ind[mol] + sysMap.nBEAD[mol]]
            z = zCG[ind[mol]:ind[mol] + sysMap.nBEAD[mol]]
            resId = allMOLid[ind[mol]:ind[mol] + sysMap.nBEAD[mol]]
            atId = allBEADid[ind[mol]:ind[mol] + sysMap.nBEAD[mol]]
            resName = allMOLname[ind[mol]:ind[mol] + sysMap.nBEAD[mol]]
            atName = allBEADname[ind[mol]:ind[mol] + sysMap.nBEAD[mol]]
            molPdbName = outPath + 'CGmol_' + sysMap.MOL[mol] + '.pdb'
            print('   CG molecular structure: {}'.format(molPdbName))
            MolConf = _cTRAJ(molPdbName, 'w')
            MolConf.time = CGtraj.time
            MolConf.nAtom = len(x)
            MolConf.resId = resId
            MolConf.resName = resName
            MolConf.atId = atId
            MolConf.atName = atName
            MolConf.boxMat = boxMatAv
            MolConf.x = x
            MolConf.y = y
            MolConf.z = z
            MolConf.write(modelN=1, sysName=sysMap.MOL[mol])
            MolConf.close()
    if not readTRJ:
        print('output file:')
    with open(outTop, 'w') as topFid:
        print('   CG topology: ' + outTop + '\n')
        topFid.write('MOL =  ')
        for mol in range(len(sysMap.MOL)):
            topFid.write('{}  '.format(sysMap.MOL[mol]))
        topFid.write('\n')
        topFid.write('NMOL =  ')
        for mol in range(len(sysMap.MOL)):
            topFid.write('{}  '.format(sysMap.nMOL[mol]))
        if _np.max(sysMap.nDIHEDRALtype > 0):
            topFid.write('\n\nEXCL = {}'.format(3))
        elif _np.max(sysMap.nANGLEtype > 0):
            topFid.write('\n\nEXCL = {}'.format(2))
        elif _np.max(sysMap.nBONDtype > 0):
            topFid.write('\n\nEXCL = {}'.format(1))
        else:
            topFid.write('\n\nEXCL = {}'.format(0))
        topFid.write('\n\n########################################\n########################################\n')
        for mol in range(len(sysMap.MOL)):
            topFid.write('\nMOL:' + sysMap.MOL[mol] + ':BEAD = ' + str(sysMap.nBEAD[mol]) + '\n')
            topFid.write('# BeadId  BeadName  BeadType  LmpType  BeadCharge   BeadMass\n')
            for i in range(sysMap.nBEAD[mol]):
                if coulomb:
                    q = sysMap.BEADch[mol][i]
                else:
                    q = 0.0
                topFid.write('  {0:^6d}  {1:^8s}  {2:^8s}  {3:^7d}  {4:^10.3f}   {5:^10.3f}\n'.format(i + 1,
                    sysMap.BEADname[mol][i], sysMap.BEADtype[mol][i], sysMap.LMPtype[mol][i], q, sysMap.BEADms[mol][i]))
            if sysMap.nBONDtype[mol] > 0:
                topFid.write('\nMOL:' + sysMap.MOL[mol] + ':BOND = ' + str(sysMap.nBONDtype[mol]) + '\n')
                topFid.write('# BondTypeId   BondName : bead1 bead2 [, ...]\n')
                for i in range(sysMap.nBONDtype[mol]):
                    idx = sysMap.BONDtype[mol][i] + 1
                    A = sysMap.BONDtypeName[mol][i, 0]
                    B = sysMap.BONDtypeName[mol][i, 1]
                    topFid.write('{0:^5d}{1:^11s}  :  '.format(idx, A + '-' + B))
                    for j in range(len(sysMap.BONDtypeIdx[mol][i])):
                        ind = sysMap.BONDtypeIdx[mol][i][j] + 1
                        topFid.write('{0:d} {1:d}'.format(ind[0], ind[1]))
                        if j != (len(sysMap.BONDtypeIdx[mol][i]) - 1):
                            topFid.write(', ')
                    topFid.write('\n')
            else:
                topFid.write('\nMOL:' + sysMap.MOL[mol] + ':BOND = 0\n')
            if sysMap.nANGLEtype[mol] > 0:
                topFid.write('\nMOL:' + sysMap.MOL[mol] + ':ANGLE = ' + str(sysMap.nANGLEtype[mol]) + '\n')
                topFid.write('# AngleTypeId   AngleName : bead1 bead2 bead3 [, ...]\n')
                for i in range(sysMap.nANGLEtype[mol]):
                    idx = sysMap.ANGLEtype[mol][i] + 1
                    A = sysMap.ANGLEtypeName[mol][i, 0]
                    B = sysMap.ANGLEtypeName[mol][i, 1]
                    C = sysMap.ANGLEtypeName[mol][i, 2]
                    topFid.write('{0:^5d}{1:^17s}  :  '.format(idx, A + '-' + B + '-' + C))
                    for j in range(len(sysMap.ANGLEtypeIdx[mol][i])):
                        ind = sysMap.ANGLEtypeIdx[mol][i][j] + 1
                        topFid.write('{0:d} {1:d} {2:d}'.format(ind[0], ind[1], ind[2]))
                        if j != (len(sysMap.ANGLEtypeIdx[mol][i]) -1):
                            topFid.write(', ')
                    topFid.write('\n')
            else:
                topFid.write('\nMOL:' + sysMap.MOL[mol] + ':ANGLE = 0\n')
            if sysMap.nDIHEDRALtype[mol] > 0:
                topFid.write('\nMOL:' + sysMap.MOL[mol] + ':DIHEDRAL = ' + str(sysMap.nDIHEDRALtype[mol]) + '\n')
                topFid.write('# DihedralTypeId   DihedralName : bead1 bead2 bead3 bead4 [, ...]\n')
                for i in range(sysMap.nDIHEDRALtype[mol]):
                    idx = sysMap.DIHEDRALtype[mol][i] + 1
                    A = sysMap.DIHEDRALtypeName[mol][i, 0]
                    B = sysMap.DIHEDRALtypeName[mol][i, 1]
                    C = sysMap.DIHEDRALtypeName[mol][i, 2]
                    D = sysMap.DIHEDRALtypeName[mol][i, 3]
                    topFid.write('{0:^5d}{1:^23s}  :  '.format(idx, A + '-' + B + '-' + C + '-' + D))
                    for j in range(len(sysMap.DIHEDRALtypeIdx[mol][i])):
                        ind = sysMap.DIHEDRALtypeIdx[mol][i][j] + 1
                        topFid.write('{0:d} {1:d} {2:d} {3:d}'.format(ind[0], ind[1], ind[2], ind[3]))
                        if j != (len(sysMap.DIHEDRALtypeIdx[mol][i]) -1):
                            topFid.write(', ')
                    topFid.write('\n')
            else:
                topFid.write('\nMOL:' + sysMap.MOL[mol] + ':DIHEDRAL = 0\n')
            topFid.write('\n########################################\n########################################\n')

#####################################################################
#####################################################################
#####################################################################

def histCalc(inTraj, top, outTab=None, RMaxNB=20, RMaxB=10,
             BinNB=0.2, BinB=0.02, BinA=2, BinD=5, normalizeBonded=False):
    '''This program calculates all radial and bonded distribution functions of the system from the CG trajectory based on the topology file.

Parameters:
• inTraj: Input CG trajectory path.
• top: The CG topology file.
• outTab: The name of output distribution file. The default filename is ’CGsystem’.
• RMaxNB: The maximum distance used in the calculation of the radial distribution functions. The default is 20 Angstrom.
• RMaxB: The maximum distance used in the calculation of the bond distribution functions. The default is 10 Angstrom.
• BinNB: The bin width used in the calculation of the radial distribution functions. The default is 0.2 Angstrom.
• BinB: The bin width used in the calculation of the bond distribution functions. The default is 0.02 Angstrom.
• BinA: The bin width used in the calculation of the angular distribution functions. The default is 2 degrees.
• BinD: The bin width used in the calculation of the dihedral distribution functions. The default is 5 degrees.
• normalizeBonded: if True, all bonded histograms are normalized to one. The default is False.

Output:
• System’s distribution functions ([outTab].hist if normalizeBonded=False or [outTab].dist if normalizeBonded=True)'''

    ##########################################
    ##########################################

    if outTab == None:
        if normalizeBonded:
            outTab = 'CGsystem' + '.dist'
        else:
            outTab = 'CGsystem' + '.hist'
    else:
        outTab = _Rstrip(outTab, ['.dist', '.hist'])
        if normalizeBonded:
            outTab = outTab + '.dist'
        else:
            outTab = outTab + '.hist'
    CGtraj = _cTRAJ(inTraj, 'r')
    sysTop = _cTOP(top)
    sysTop.SetBondedIndex()
    sysTop.SetNonBondedIndex()

    nNB = len(sysTop.NonBONDED_Set)
    npoint = _np.int(_np.round(RMaxNB / BinNB)) + 1
    xRDF = _np.linspace(0, RMaxNB, npoint, dtype=float)
    jacob = (4. / 3) * _np.pi * (xRDF[1:] ** 3 - xRDF[0:-1] ** 3)
    RDF = _np.zeros([nNB, len(xRDF) - 1], dtype=float)

    nB = len(sysTop.BONDtypeSet)
    npoint = _np.int(_np.round(RMaxB / BinB)) + 1
    xBDF = _np.linspace(0, RMaxB, npoint, dtype=float)
    BDF = _np.zeros([nB, len(xBDF) - 1], dtype=float)

    nA = len(sysTop.ANGLEtypeSet)
    npoint = _np.int(_np.round(180 / BinA)) + 1
    xADF = _np.linspace(0, 180, npoint, dtype=float)
    ADF = _np.zeros([nA, len(xADF) - 1], dtype=float)

    nD = len(sysTop.DIHEDRALtypeSet)
    npoint = _np.int(_np.round(360 / BinD)) + 1
    xDDF = _np.linspace(-180, 180, npoint, dtype=float)
    DDF = _np.zeros([nD, len(xDDF) - 1], dtype=float)

    sysDFs = _cDFs()
    nSnap = 0
    while True:
        CGtraj.read(nAtom=sum(sysTop.nMOL * sysTop.nBEAD))
        if CGtraj.eof:
            CGtraj.close()
            break
        CGtraj.scaleSnap()
        volume = CGtraj.boxMat[0] * CGtraj.boxMat[1] * CGtraj.boxMat[2]
        if CGtraj.xu is not None:
            x, y, z = CGtraj.xu, CGtraj.yu, CGtraj.zu
        elif CGtraj.x is not None:
            x, y, z = CGtraj.x, CGtraj.y, CGtraj.z
        else:
            mess = 'Error in parsing the trajectory'
            raise _scg4pyError(mess)
        pairData = sysDFs.calcNB(CGtraj.xs, CGtraj.ys, CGtraj.zs, sysTop.NonBONDED_SetIdx, CGtraj.boxMat)
        bondData = sysDFs.calcB(x, y, z, sysTop.BONDtypeSetIdx)
        AngleData = sysDFs.calcA(x, y, z, sysTop.ANGLEtypeSetIdx)
        DihedralData = sysDFs.calcD(x, y, z, sysTop.DIHEDRALtypeSetIdx)
        for i in range(nNB):
            Hist = _np.histogram(pairData[i], bins=xRDF)[0]
            nIntract = len(sysTop.NonBONDED_SetIdx[i])
            RDF[i] = RDF[i] + (volume / nIntract) * (Hist / jacob)
        for i in range(nB):
            Hist = _np.histogram(bondData[i], bins=xBDF)[0]
            BDF[i] = BDF[i] + Hist
        for i in range((nA)):
            Hist = _np.histogram(AngleData[i], bins=xADF)[0]
            ADF[i] = ADF[i] + Hist
        for i in range((nD)):
            Hist = _np.histogram(DihedralData[i], bins=xDDF)[0]
            DDF[i] = DDF[i] + Hist
        nSnap += 1
        if (nSnap - 1) % 5 == 0:
            if CGtraj.time:
                print('\rTime = {}'.format(CGtraj.time), end='')
            elif CGtraj.timeStep:
                print('\rTimeStep = {}'.format(CGtraj.timeStep), end='')
        if nSnap == 1:
            print()
    if CGtraj.time:
        print('\rTime = {}'.format(CGtraj.time))
    elif CGtraj.timeStep:
        print('\rTimeStep = {}'.format(CGtraj.timeStep))
    print('\nnumber of snapshots = {}'.format(nSnap))
    del CGtraj
    RDF /= nSnap
    BDF /= nSnap
    ADF /= nSnap
    DDF /= nSnap
    xRDFi = xRDF[0:-1] + 0.5 * BinNB
    xBDFi = xBDF[0:-1] + 0.5 * BinB
    xADFi = xADF[0:-1] + 0.5 * BinA
    xDDFi = xDDF[0:-1] + 0.5 * BinD

    xRDF, yRDF = _np.zeros(nNB, dtype=object), _np.zeros(nNB, dtype=object)
    xBDF, yBDF = _np.zeros(nB, dtype=object), _np.zeros(nB, dtype=object)
    xADF, yADF = _np.zeros(nA, dtype=object), _np.zeros(nA, dtype=object)
    xDDF, yDDF = _np.zeros(nD, dtype=object), _np.zeros(nD, dtype=object)
    sysTab = _cTAB(outTab, 'w')
    for i in range(nNB):
        xRDF[i] = xRDFi
        yRDF[i] = RDF[i]
    for i in range(nB):
        X, Y = sysTab.trim(xBDFi, BDF[i], yMin=1e-5, BothSide=True)
        xBDF[i] = X
        if normalizeBonded:
            yBDF[i] = Y / _np.trapz(Y, dx=BinB)
        else:
            yBDF[i] = Y
    for i in range(nA):
        X, Y = sysTab.trim(xADFi, ADF[i], yMin=1e-5, BothSide=True)
        xADF[i] = X
        if normalizeBonded:
            yADF[i] = Y / _np.trapz(Y, dx=BinA)
        else:
            yADF[i] = Y
    for i in range(nD):
        xDDF[i] = xDDFi
        Y = DDF[i]
        if normalizeBonded:
            yDDF[i] = Y / _np.trapz(Y, dx=BinD)
        else:
            yDDF[i] = Y
    sysTab.setattr(NonBondType=sysTop.NonBONDED_Set, NonBondX=xRDF, NonBondY=yRDF, BondName=sysTop.BONDtypeSetName,
                   BondType=sysTop.BONDtypeSet, BondX=xBDF, BondY=yBDF, AngleName=sysTop.ANGLEtypeSetName,
                   AngleType=sysTop.ANGLEtypeSet, AngleX=xADF, AngleY=yADF, DihedralName=sysTop.DIHEDRALtypeSetName,
                   DihedralType=sysTop.DIHEDRALtypeSet, DihedralX=xDDF, DihedralY=yDDF)
    print('output : {}'.format(outTab))
    sysTab.write()

#####################################################################
#####################################################################
#####################################################################

def histTrim(inTabs, outTab=None):
    '''This program exerts some modifications on distribution functions by smoothing and trimming the edge of tables in order to produce the reference distribution functions.

Parameters:
• inTabs: Input distribution functions file(s). If a list of names is entered, you can choose the best distribution function from them.
• outTab: Output distribution functions file.

Output:
• Modified distribution functions file ([outTab].hist or [outTab].dist)'''

    ###############################################
    ###############################################

    def DFselecting(xi, yi, ith, Name, DFtype, nInTabs, colorId, legend):
        xlabel = 'r [Angstrom]'
        ylabel = 'Distribution'
        if DFtype == 'RDF':
            ylabel = 'RDF'
            Name = 'Non-Bonded: ' + str(Name)
        elif DFtype == 'Bond':
            Name = 'Bond: ' + str(Name[ith])
        elif DFtype == 'Angle':
            xlabel = 'theta [degree]'
            Name = 'Angle: ' + str(Name[ith])
        elif DFtype == 'Dihedral':
            xlabel = 'phi [degree]'
            Name = 'Dihedral: ' + str(Name[ith])
        if nInTabs == 1:
            x_out = xi[0][ith]
            y_out = yi[0][ith]
            return x_out, y_out
        plotNum = 0
        while True:
            _plt.subplot(121)
            for n in range(nInTabs):
                x = xi[n][ith]
                y = yi[n][ith]
                if nInTabs < 11:
                    _plt.plot(x, y, color=colorId[n])
                else:
                    _plt.plot(x, y, color=colorId[3 * n:3 * n + 3])
            _plt.xlabel(xlabel, fontsize='large')
            _plt.ylabel(ylabel, fontsize='large')
            _plt.ylim(0)
            _plt.title(Name)
            _plt.legend(legend)
            _plt.subplot(122)
            x = xi[plotNum][ith]
            bw = _np.round(_np.mean(_np.diff(x)), 4)
            y = yi[plotNum][ith]
            _plt.plot(x, y, color='b')
            _plt.xlabel(xlabel, fontsize='large')
            _plt.ylim(0)
            _plt.title('Bin Width: ' + str(bw))
            _plt.legend(str(plotNum + 1))
            _plt.show(block=False)
            _plt.pause(0.01)
            order = input('\n"s": save the selected distribution:\n')
            _plt.clf()
            if order.upper() == 'S':
                x_out = xi[plotNum][ith]
                y_out = yi[plotNum][ith]
                break
            elif order.isdigit():
                if 1 <= int(order) <= nInTabs:
                    plotNum = int(order) - 1
        return x_out, y_out

    def DFsmoothing(x, y, DFtype):
        xlabel = 'r [Angstrom]'
        ylabel = 'Distribution'
        if DFtype == 'RDF':
            ylabel = 'RDF'
        elif DFtype == 'Angle':
            xlabel = 'theta [degree]'
        elif DFtype == 'Dihedral':
            xlabel = 'phi [degree]'
        sig = 0.0
        while True:
            if sig > 0:
                SIG = sig * _np.mean(_np.diff(x))
                yy = _np.zeros(_np.size(x), dtype=float)
                for i in _np.arange(_np.size(yy)):
                    expX = _np.exp((-1 * (x[i] - x) ** 2) / (2 * SIG ** 2))
                    Z = _np.sum(expX)
                    yy[i] = _np.sum(y * expX) / Z
            else:
                yy = y
            minX = _np.min(x)
            maxX = _np.max(x)
            minY = _np.min([_np.min(y), _np.min(yy)])
            maxY = _np.max([_np.max(y), _np.max(yy)])
            _plt.subplot(121)
            _plt.plot(x, y)
            _plt.xlabel(xlabel, fontsize='large')
            _plt.ylabel(ylabel, fontsize='large')
            _plt.xlim(minX, maxX)
            _plt.ylim(minY, maxY)
            _plt.title('Main', fontsize='large')
            _plt.subplot(122)
            _plt.plot(x, yy)
            _plt.xlabel(xlabel, fontsize='large')
            _plt.xlim(minX, maxX)
            _plt.ylim(minY, maxY)
            _plt.title('Sigma= ' + str(round(sig, 4)), fontsize='large')
            _plt.show(block=False)
            _plt.pause(0.01)
            order = input('\nenter new sigma or'
                              '\n"s": save the smoothed distribution'
                              '\n"q": quit.\n')
            _plt.clf()
            if order.upper() == 'S':
                y_out = yy
                break
            elif order.upper() == 'Q':
                y_out = y
                break
            elif not _isfloat(order):
                continue
            else:
                sig = float(order)
        return y_out

    def DFtrimming(x, y, Name, DFtype):
        xlabel = 'r [Angstrom]'
        ylabel = 'Distribution'
        BothSide = True
        if DFtype == 'RDF':
            ylabel = 'RDF'
            Name = 'Non-Bonded: ' + Name[0] + '-' + Name[1]
            BothSide = False
        elif DFtype == 'Bond':
            Name = 'Bond: ' + Name[0] + '-' + Name[1]
        elif DFtype == 'Angle':
            xlabel = 'theta [degree]'
            Name = 'Angle: ' + Name[0] + '-' + Name[1] + '-' + Name[2]
        yThreshold = 1e-10
        while True:
            I = 0
            for i in range(len(y)):
                if y[i] > yThreshold:
                    I = i
                    break
            J = len(y) - 1
            if BothSide:
                for j in range(len(y) - 1, 0, -1):
                    if y[j] > yThreshold:
                        J = j
                        break
                if J == len(y) - 1:
                    X = x[I:]
                    Y = y[I:]
                else:
                    J += 1
                    X = x[I:J]
                    Y = y[I:J]
            else:
                X = x[I:]
                Y = y[I:]
            _plt.plot(x, y, 'r3', X, Y, 'b')
            _plt.xlabel(xlabel, fontsize='large')
            _plt.ylabel(ylabel, fontsize='large')
            _plt.title(Name, fontsize='large')
            _plt.show(block=False)
            _plt.pause(0.01)
            print ('\nmin of dist before trimming = ' + str(_np.round(_np.min(y), 6)))
            print ('min of dist after trimming = ' + str(_np.round(_np.min(Y), 6)))
            order = input('enter new threshold or "s" to save:\n')
            _plt.clf()
            if order.upper() == 'S':
                x_out = X
                y_out = Y
                break
            elif not _isfloat(order):
                continue
            else:
                yThreshold = float(order)
        return x_out, y_out

    ###############################################
    ###############################################

    if isinstance(inTabs, list):
        pass
    else:
        inTabs = [inTabs]
    nInTabs = len(inTabs)

    Hist = Dist = False
    if inTabs[0].endswith('.hist'):
        Hist = True
    elif inTabs[0].endswith('.dist'):
        Dist = True
    else:
        print('input file(s) must have one of ".dist" or ".hist" extensions.')
        return
    if Hist:
        for f in inTabs:
            Hist = Hist and f.endswith('.hist')
        if not Hist:
            print('input files are not compatible with each other.')
            return
    elif Dist:
        for f in inTabs:
            Dist = Dist and f.endswith('.dist')
        if not Dist:
            print('input files are not compatible with each other.')
            return

    if outTab == None:
        if Hist:
            outTab = 'CGsystem.ref.hist'
        elif Dist:
            outTab = 'CGsystem.ref.dist'
    else:
        outTab = _Rstrip(outTab, ['.hist', '.dist', '.ref'])
        if Hist:
            outTab = outTab + '.ref.hist'
        elif Dist:
            outTab = outTab + '.ref.dist'

    legend = [str(i + 1) for i in range(nInTabs)]
    colorL = _np.array([['blue'],
                          ['blue', 'red'],
                          ['blue', 'green', 'red'],
                          ['blue', 'green', 'orange', 'red'],
                          ['blue', 'm', 'green', 'orange', 'red'],
                          ['blue', 'm', 'cyan', 'green', 'orange', 'red'],
                          ['blue', 'm', 'cyan', 'lime', 'green', 'orange', 'red'],
                          ['blue', 'm', 'cyan', 'lime', 'gold', 'green', 'orange', 'red'],
            ['blue', 'm', 'cyan', 'dodgerblue', 'lime', 'gold', 'green', 'orange', 'red'],
    ['blue', 'm', 'cyan', 'dodgerblue', 'lime', 'gold', 'green', 'orange', 'chocolate', 'red']])
    if nInTabs < 11:
        colorId = colorL[nInTabs - 1]
    else:
        colorId = _np.random.rand(nInTabs * 3, 1)

    sysTab = _cTAB(inTabs[0], 'r')
    BondType = sysTab.BondType
    BondName = sysTab.BondName
    AngleType = sysTab.AngleType
    AngleName = sysTab.AngleName
    DihedralType = sysTab.DihedralType
    DihedralName = sysTab.DihedralName
    NonBondType = sysTab.NonBondType
    nB = len(BondType)
    nA = len(AngleType)
    nD = len(DihedralType)
    nNB = len(NonBondType)
    xB = _np.zeros(nInTabs, dtype=object)
    yB = _np.zeros(nInTabs, dtype=object)
    xA = _np.zeros(nInTabs, dtype=object)
    yA = _np.zeros(nInTabs, dtype=object)
    xD = _np.zeros(nInTabs, dtype=object)
    yD = _np.zeros(nInTabs, dtype=object)
    xNB = _np.zeros(nInTabs, dtype=object)
    yNB = _np.zeros(nInTabs, dtype=object)

    for n in range(nInTabs):
        sysTab = _cTAB(inTabs[n], 'r')
        if not (_np.all(BondName == sysTab.BondName) and _np.all(BondType == sysTab.BondType)):
            mess = 'Bond terms of the "{}" file are not complible with others'.format(inTabs[n])
            raise _scg4pyError(mess)
        if not (_np.all(AngleName == sysTab.AngleName) and _np.all(AngleType == sysTab.AngleType)):
            mess = 'Angle terms of the "{}" file are not complible with others'.format(inTabs[n])
            raise _scg4pyError(mess)
        if not (_np.all(DihedralName == sysTab.DihedralName) and _np.all(DihedralType == sysTab.DihedralType)):
            mess = 'Dihedral terms of the "{}" file are not complible with others'.format(inTabs[n])
            raise _scg4pyError(mess)
        if not _np.all(NonBondType == sysTab.NonBondType):
            mess = 'Non-Bonded terms of the "{}" file are not complible with others'.format(inTabs[n])
            raise _scg4pyError(mess)
        xB[n] = sysTab.BondX
        yB[n] = sysTab.BondY
        xA[n] = sysTab.AngleX
        yA[n] = sysTab.AngleY
        xD[n] = sysTab.DihedralX
        yD[n] = sysTab.DihedralY
        xNB[n] = sysTab.NonBondX
        yNB[n] = sysTab.NonBondY
    xB_ref = _np.zeros(nB, dtype=object)
    yB_ref = _np.zeros(nB, dtype=object)
    xA_ref = _np.zeros(nA, dtype=object)
    yA_ref = _np.zeros(nA, dtype=object)
    xD_ref = _np.zeros(nD, dtype=object)
    yD_ref = _np.zeros(nD, dtype=object)
    xNB_ref = _np.zeros(nNB, dtype=object)
    yNB_ref = _np.zeros(nNB, dtype=object)
    # _plt.rc('font', family='monospace')
    for i in range(nNB):
        print ('\n\n\n##################################################')
        DFname = 'Non-Bonded_' + NonBondType[i, 0] + '-' + NonBondType[i, 1]
        print ('\n### RDF selecting {0:d}/{1:d}: {2:s}'.format(i + 1, nNB, DFname))
        x, y = DFselecting(xNB, yNB, i, i + 1, 'RDF', nInTabs, colorId, legend)
        print('\n### RDF smoothing {0:d}/{1:d}: {2:s}'.format(i + 1, nNB, DFname))
        y_smooth = DFsmoothing(x, y, 'RDF')
        print('\n### RDF trimming {0:d}/{1:d}: {2:s}'.format(i + 1, nNB, DFname))
        X, Y = DFtrimming(x, y_smooth, NonBondType[i], 'RDF')
        xNB_ref[i] = X
        yNB_ref[i] = Y
    for i in range(nB):
        print ('\n\n\n##################################################')
        DFname = 'Bond_' + BondName[i, 0] + '-' + BondName[i, 1]
        print('\n### BDF selecting {0:d}/{1:d}: {2:s}'.format(i + 1, nB, DFname))
        x, y = DFselecting(xB, yB, i, BondType + 1,'Bond', nInTabs, colorId, legend)
        print('\n### BDF smoothing {0:d}/{1:d}: {2:s}'.format(i + 1, nB, DFname))
        y_smooth = DFsmoothing(x, y, 'Bond')
        print('\n### BDF trimming {0:d}/{1:d}: {2:s}'.format(i + 1, nB, DFname))
        X, Y = DFtrimming(x, y_smooth, BondName[i] , 'Bond')
        xB_ref[i] = X
        if Dist:
            yB_ref[i] = Y / _np.trapz(Y, x=X)
        elif Hist:
            yB_ref[i] = Y
    for i in range(nA):
        print('\n\n\n##################################################')
        DFname = 'Angle_' + AngleName[i, 0] + '-' + AngleName[i, 1] + '-' + AngleName[i, 2]
        print('\n### ADF selecting {0:d}/{1:d}: {2:s}'.format(i + 1, nA, DFname))
        x, y = DFselecting(xA, yA, i, AngleType + 1, 'Angle', nInTabs, colorId, legend)
        print('\n### ADF smoothing {0:d}/{1:d}: {2:s}'.format(i + 1, nA, DFname))
        y_smooth = DFsmoothing(x, y, 'Angle')
        print('\n### ADF trimming {0:d}/{1:d}: {2:s}'.format(i + 1, nA, DFname))
        X, Y = DFtrimming(x, y_smooth, AngleName[i], 'Angle')
        xA_ref[i] = X
        if Dist:
            yA_ref[i] = Y / _np.trapz(Y, x=X)
        elif Hist:
            yA_ref[i] = Y
    for i in range(nD):
        print('\n\n\n##################################################')
        DFname = 'Dihedral_' + DihedralName[i, 0] + '-' + DihedralName[i, 1] + '-' + DihedralName[i, 2] + '-' + DihedralName[i, 3]
        print('\n### DDF trimming {0:d}/{1:d}: {2:s}'.format(i + 1, nD, DFname))
        x, y = DFselecting(xD, yD, i, DihedralType + 1, 'Dihedral', nInTabs, colorId, legend)
        print('\n### DDF smoothing {0:d}/{1:d}: {2:s}'.format(i + 1, nD, DFname))
        y_smooth = DFsmoothing(x, y, 'Dihedral')
        xD_ref[i] = x
        if Dist:
            yD_ref[i] = y_smooth / _np.trapz(y_smooth, x=x)
        elif Hist:
            yD_ref[i] = y_smooth
    _plt.close()
    print('\noutput : {}'.format(outTab))
    sysTab = _cTAB(outTab, 'w')
    sysTab.setattr(NonBondType=NonBondType, NonBondX=xNB_ref, NonBondY=yNB_ref, BondName=BondName,
                   BondType=BondType, BondX=xB_ref, BondY=yB_ref, AngleName=AngleName,
                   AngleType=AngleType, AngleX=xA_ref, AngleY=yA_ref, DihedralName=DihedralName,
                   DihedralType=DihedralType, DihedralX=xD_ref, DihedralY=yD_ref)
    sysTab.write()

#####################################################################
#####################################################################
#####################################################################

def hist2pot(inTab, outPot=None, initPot=True, T=300):
    '''This program produces the initial CG potential from the reference distribution functions through direct Boltzmann inversion.

Parameters:
• inTab: Input reference distribution function file.
• outPot: Output potential file. The default is the same as the input filename.
• initPot: If True, the non-bonded potentials are weakened to eliminate the bad effects of the initial potentials. The default is True.
• T: The temperature of the system. The default is 300 Kelvin.

Output:
• Initial potential functions ([outPot].pot)'''

    ##########################################
    ##########################################

    kB = 1.987204e-03
    inTab = str(inTab)
    if inTab.endswith('.hist') or inTab.endswith('.dist'):
        pass
    else:
        mess = 'input file must be one of ".dist" or ".hist".'
        raise _scg4pyError(mess)

    if outPot == None:
        outPot = _Rstrip(inTab, ['.hist', '.dist', '.ref']) + '.pot'
    else:
        outPot = _Rstrip(outPot, '.pot') + '.pot'

    ##########################################

    def genPot(x, y, name, DFtype, kB, T):
        ylabel = 'U (kcal / mol)'
        xlabel = 'r [Angstrom]'
        if DFtype == 'RDF':
            pass
        elif DFtype == 'Bond':
            pass
        elif DFtype == 'Angle':
            xlabel = 'theta [degree]'
        elif DFtype == 'Dihedral':
            xlabel = 'phi [degree]'
        sig = 0.5
        while True:
            if sig > 0:
                SIG = sig * _np.mean(_np.diff(x))
                yy = _np.zeros(_np.size(x), dtype=float)
                for i in _np.arange(_np.size(yy)):
                    expX = _np.exp((-1 * (x[i] - x) ** 2) / (2 * SIG ** 2))
                    Z = _np.sum(expX)
                    yy[i] = _np.sum(y * expX) / Z
            elif sig == 0:
                yy = y

            minX = _np.min(x)
            maxX = _np.max(x)
            yPot = -1 * kB * T * _np.log(y)
            if sig < 0 :
                yyPot = _np.zeros(len(y), dtype=float)
            else:
                yyPot = -1 * kB * T * _np.log(yy)
            minY = _np.min([_np.min(yPot), _np.min(yyPot)])
            maxY = _np.max([_np.max(yPot), _np.max(yyPot)])

            _plt.subplot(121)
            _plt.plot(x, yPot)
            _plt.xlabel(xlabel, fontsize='large')
            _plt.ylabel(ylabel, fontsize='large')
            _plt.xlim(minX, maxX)
            _plt.ylim(minY, maxY)
            _plt.title(name, fontsize='large')
            _plt.subplot(122)
            _plt.plot(x, yyPot)
            _plt.xlabel(xlabel, fontsize='large')
            _plt.xlim(minX, maxX)
            _plt.ylim(minY, maxY)
            _plt.title('Sigma= ' + str(sig), fontsize='large')
            _plt.show(block=False)
            _plt.pause(0.01)

            order = input('enter new sigma or\n"s" to save\n"q" to quit\n')
            _plt.clf()
            if order.upper() == 'S':
                pot_out = yyPot
                break
            elif order.upper() == 'Q':
                pot_out = yPot
                break
            elif not _isfloat(order):
                continue
            else:
                sig = float(order)
        return pot_out

    ########################################

    sysTab = _cTAB(inTab, 'r')
    nNB = len(sysTab.NonBondType)
    nB = len(sysTab.BondType)
    nA = len(sysTab.AngleType)
    nD = len(sysTab.DihedralType)
    yB_pot = _np.zeros(nB, dtype=object)
    yA_pot = _np.zeros(nA, dtype=object)
    yD_pot = _np.zeros(nD, dtype=object)
    yNB_pot = _np.zeros(nNB, dtype=object)
    for i in range(nNB):
        print('\n\n\n##################################################')
        DFname = 'Non-Bonded_' + sysTab.NonBondType[i, 0] + '-' + sysTab.NonBondType[i, 1]
        print('\n### Potential {0:d}/{1:d}: {2:s}'.format(i + 1, nNB, DFname))
        x, y = sysTab.NonBondX[i], sysTab.NonBondY[i]
        y[y <= 1e-10] = 1e-10
        yPot = genPot(x, y, DFname, 'RDF', kB, T)
        if initPot:
            scale = 0.1
            n = 0
            while n < len(yPot) and yPot[n] > 0:
                n += 1
            if n < (len(yPot) - 1):
                yPot[n:] = scale * yPot[n:]
        yNB_pot[i] = yPot
    for i in range(nB):
        print('\n\n\n##################################################')
        DFname = 'Bond_' + sysTab.BondName[i, 0] + '-' + sysTab.BondName[i, 1]
        print('\n### Potential {0:d}/{1:d}: {2:s}'.format(i + 1, nB, DFname))
        x, y = sysTab.BondX[i], sysTab.BondY[i]
        y = y / (x ** 2)
        y[y <= 1e-10] = 1e-10
        yPot = genPot(x, y, DFname, 'BDF', kB, T)
        yB_pot[i] = yPot - _np.min(yPot)
    for i in range(nA):
        print('\n\n\n##################################################')
        DFname = 'Angle_' + sysTab.AngleName[i, 0] + '-' + sysTab.AngleName[i, 1] + '-' + sysTab.AngleName[i, 2]
        print('\n### Potential {0:d}/{1:d}: {2:s}'.format(i + 1, nA, DFname))
        x, y = sysTab.AngleX[i], sysTab.AngleY[i]
        y = y / (_np.sin(_np.deg2rad(x)))
        y[y <= 1e-10] = 1e-10
        yPot = genPot(x, y, DFname, 'ADF', kB, T)
        yA_pot[i] = yPot - _np.min(yPot)
    for i in range(nD):
        print('\n\n\n##################################################')
        DFname = 'Dihedral_' + sysTab.DihedralName[i, 0] + '-' + sysTab.DihedralName[i, 1] + \
                 '-' + sysTab.DihedralName[i, 2] + '-' + sysTab.DihedralName[i, 3]
        print('\n### Potential {0:d}/{1:d}: {2:s}'.format(i + 1, nD, DFname))
        x, y = sysTab.DihedralX[i], sysTab.DihedralY[i]
        y[y <= 1e-10] = 1e-10
        yPot = genPot(x, y, DFname, 'DDF', kB, T)
        yD_pot[i] = yPot - _np.min(yPot)
    _plt.close()
    print('\noutput potential: {}'.format(outPot))
    sysPot = _cTAB(outPot, 'w')
    sysPot.setattr(NonBondType=sysTab.NonBondType, NonBondX=sysTab.NonBondX, NonBondY=yNB_pot, BondName=sysTab.BondName,
                   BondType=sysTab.BondType, BondX=sysTab.BondX, BondY=yB_pot, AngleName=sysTab.AngleName,
                   AngleType=sysTab.AngleType, AngleX=sysTab.AngleX, AngleY=yA_pot, DihedralName=sysTab.DihedralName,
                   DihedralType=sysTab.DihedralType, DihedralX=sysTab.DihedralX, DihedralY=yD_pot)
    sysPot.write()

#####################################################################
#####################################################################
#####################################################################

def plotTab(inTabs, legend=-1):
    '''This program stores the figures of both the distribution function tables and potential function tables.

Parameters:
• inTabs: List of the input tables.
• legend: List of the legend names. If -1, The legends are named from number 1.

Output:
• The figure saved for each table (*.png)'''

    ##########################################
    ##########################################

    if isinstance(inTabs, list):
        pass
    else:
        inTabs = [inTabs]
    nInTabs = len(inTabs)

    if legend == -1:
        legend = [str(i + 1) for i in range(nInTabs)]
    elif isinstance(legend, list):
        pass
    else:
        legend = [legend]
    nLeg = len(legend)

    isPOT = _np.zeros(nInTabs, dtype=bool)
    isDPOT = _np.zeros(nInTabs, dtype=bool)
    isHIST = _np.zeros(nInTabs, dtype=bool)
    isDIST = _np.zeros(nInTabs, dtype=bool)
    for i in range(nInTabs):
        isPOT[i] = inTabs[i].endswith('.pot')
        isDPOT[i] = inTabs[i].endswith('.dpot')
        isHIST[i] = inTabs[i].endswith('.hist')
        isDIST[i] = inTabs[i].endswith('.dist')

    if _np.all(isPOT) == True:
        fType = 'POT'
    elif _np.all(isDPOT) == True:
        fType = 'DPOT'
    elif _np.all(isHIST) == True:
        fType = 'HIST'
    elif _np.all(isDIST) == True:
        fType = 'DIST'
    else:
        mess = 'error in reading input files.'
        raise _scg4pyError(mess)
    if nInTabs != nLeg:
        mess = "the number of input file(s) is not equal to input legend(s)."
        raise _scg4pyError(mess)
    colorL = _np.array([['blue'],
                        ['blue', 'red'],
                        ['blue', 'green', 'red'],
                        ['blue', 'green', 'orange', 'red'],
                        ['blue', 'm', 'green', 'orange', 'red'],
                        ['blue', 'm', 'cyan', 'green', 'orange', 'red'],
                        ['blue', 'm', 'cyan', 'lime', 'green', 'orange', 'red'],
                        ['blue', 'm', 'cyan', 'lime', 'gold', 'green', 'orange', 'red'],
                        ['blue', 'm', 'cyan', 'dodgerblue', 'lime', 'gold', 'green', 'orange', 'red'],
                        ['blue', 'm', 'cyan', 'dodgerblue', 'lime', 'gold', 'green', 'orange', 'chocolate', 'red']])
    if nLeg < 11:
        colorId = colorL[nLeg - 1]
    else:
        colorId = _np.random.rand(nInTabs * 3, 1)

    sysTab = _cTAB(inTabs[0], 'r')
    BondType = sysTab.BondType
    BondName = sysTab.BondName
    AngleType = sysTab.AngleType
    AngleName = sysTab.AngleName
    DihedralType = sysTab.DihedralType
    DihedralName = sysTab.DihedralName
    NonBondType = sysTab.NonBondType
    nB = len(BondType)
    nA = len(AngleType)
    nD = len(DihedralType)
    nNB = len(NonBondType)
    xB = _np.zeros(nInTabs, dtype=object)
    yB = _np.zeros(nInTabs, dtype=object)
    xA = _np.zeros(nInTabs, dtype=object)
    yA = _np.zeros(nInTabs, dtype=object)
    xD = _np.zeros(nInTabs, dtype=object)
    yD = _np.zeros(nInTabs, dtype=object)
    xNB = _np.zeros(nInTabs, dtype=object)
    yNB = _np.zeros(nInTabs, dtype=object)

    for n in range(nInTabs):
        sysTab = _cTAB(inTabs[n], 'r')
        if not (_np.all(BondName == sysTab.BondName) and _np.all(BondType == sysTab.BondType)):
            mess = 'Bond terms of the "{}" file are not complible with others'.format(inTabs[n])
            raise _scg4pyError(mess)
        if not (_np.all(AngleName == sysTab.AngleName) and _np.all(AngleType == sysTab.AngleType)):
            mess = 'Angle terms of the "{}" file are not complible with others'.format(inTabs[n])
            raise _scg4pyError(mess)
        if not (_np.all(DihedralName == sysTab.DihedralName) and _np.all(DihedralType == sysTab.DihedralType)):
            mess = 'Dihedral terms of the "{}" file are not complible with others'.format(inTabs[n])
            raise _scg4pyError(mess)
        if not _np.all(NonBondType == sysTab.NonBondType):
            mess = 'Non-Bonded terms of the "{}" file are not complible with others'.format(inTabs[n])
            raise _scg4pyError(mess)
        xB[n] = sysTab.BondX
        yB[n] = sysTab.BondY
        xA[n] = sysTab.AngleX
        yA[n] = sysTab.AngleY
        xD[n] = sysTab.DihedralX
        yD[n] = sysTab.DihedralY
        xNB[n] = sysTab.NonBondX
        yNB[n] = sysTab.NonBondY

    # _plt.rc('font', family='monospace')
    _plt.figure(figsize=(14, 7))
    for i in range(nNB):
        Name = 'Non-Bonded_{0:s}-{1:s}'.format(NonBondType[i, 0], NonBondType[i, 1])
        for n in range(nInTabs):
            x = xNB[n][i]
            y = yNB[n][i]
            if nLeg < 11:
                _plt.plot(x, y, color=colorId[n], lw=1)
            else:
                _plt.plot(x, y, color=colorId[3 * n:3 * n + 3], lw=1)
        title = Name.replace('_', ': ')
        _plt.xlabel('r (Angstrom)', fontsize=14)
        if fType == 'HIST' or fType == 'DIST':
            _plt.ylabel('RDF', fontsize=14)
        elif fType == 'POT' or fType == 'DPOT':
            _plt.ylabel('U (kcal/mol)', fontsize=14)
        _plt.autoscale(True, axis=u'both', tight=True)
        _plt.title(title)
        _plt.legend(legend)
        if fType == 'HIST' or fType == 'DIST':
            _plt.savefig(Name + '.df.png')
        elif fType == 'POT':
            _plt.savefig(Name + '.pot.png')
        elif fType == 'DPOT':
            _plt.savefig(Name + '.dpot.png')
        _plt.clf()

    for i in range(nB):
        Name = 'Bond_{0:s}-{1:s}'.format(BondName[i, 0], BondName[i, 1])
        for n in range(nInTabs):
            x = xB[n][i]
            y = yB[n][i]
            if nLeg < 11:
                _plt.plot(x, y, color=colorId[n])
            else:
                _plt.plot(x, y, color=colorId[3 * n:3 * n + 3])
        title = Name.replace('_', ': ')
        _plt.xlabel('r (Angstrom)', fontsize=14)
        if fType == 'HIST' or fType == 'DIST':
            _plt.ylabel('Distribution', fontsize=14)
        elif fType == 'POT' or fType == 'DPOT':
            _plt.ylabel('U (kcal/mol)', fontsize=14)
        _plt.title(title)
        _plt.autoscale(True, axis=u'both', tight=True)
        _plt.legend(legend)
        if fType == 'HIST' or fType == 'DIST':
            _plt.savefig(Name + '.df.png')
        elif fType == 'POT':
            _plt.savefig(Name + '.pot.png')
        elif fType == 'DPOT':
            _plt.savefig(Name + '.dpot.png')
        _plt.clf()
    for i in range(nA):
        Name = 'Angle_{0:s}-{1:s}-{2:s}'.format(AngleName[i, 0], AngleName[i, 1], AngleName[i, 2])
        for n in range(nInTabs):
            x = xA[n][i]
            y = yA[n][i]
            if nLeg < 11:
                _plt.plot(x, y, color=colorId[n])
            else:
                _plt.plot(x, y, color=colorId[3 * n:3 * n + 3])
        title = Name.replace('_', ': ')
        _plt.xlabel('theta (degree)', fontsize=14)
        if fType == 'HIST' or fType == 'DIST':
            _plt.ylabel('Distribution', fontsize=14)
        elif fType == 'POT' or fType == 'DPOT':
            _plt.ylabel('U (kcal\mol)', fontsize=14)
        _plt.autoscale(True, axis=u'both', tight=True)
        _plt.title(title)
        _plt.legend(legend)
        if fType == 'HIST' or fType == 'DIST':
            _plt.savefig(Name + '.df.png')
        elif fType == 'POT':
            _plt.savefig(Name + '.pot.png')
        elif fType == 'DPOT':
            _plt.savefig(Name + '.dpot.png')
        _plt.clf()
    for i in range(nD):
        Name = 'Dihedral_{0:s}-{1:s}-{2:s}-{3:s}'.format(DihedralName[i, 0], DihedralName[i, 1],
                                                         DihedralName[i, 2], DihedralName[i, 3])
        for n in range(nInTabs):
            x = xD[n][i]
            y = yD[n][i]
            if nLeg < 11:
                _plt.plot(x, y, color=colorId[n])
            else:
                _plt.plot(x, y, color=colorId[3 * n:3 * n + 3])
        title = Name.replace('_', ': ')
        _plt.xlabel('phi (degree)', fontsize=14)
        if fType == 'HIST' or fType == 'DIST':
            _plt.ylabel('Distribution', fontsize=14)
        elif fType == 'POT' or fType == 'DPOT':
            _plt.ylabel('U (kcal\mol)', fontsize=14)
        _plt.autoscale(True, axis=u'both', tight=True)
        _plt.title(title)
        _plt.legend(legend)
        if fType == 'HIST' or fType == 'DIST':
            _plt.savefig(Name + '.df.png')
        elif fType == 'POT':
            _plt.savefig(Name + '.pot.png')
        elif fType == 'DPOT':
            _plt.savefig(Name + '.dpot.png')
        _plt.clf()
    _plt.close()

#####################################################################
#####################################################################
#####################################################################

def genLMPdata(inConf, top, output=None):
    '''This program reads an input configuration and then generates the LAMMPS data file. Also, it produces a ’psf’ file to view the CG structure in VMD software.

Parameters:
• inConf : Input structure file. All types of ’lammpstrj’, ’pdb’, ’gro’, ’xtc’, and ’dcd’ file formats are supported.
• top: CG topology file.
• output: Output name of the LAMMPS data file and ’psf’ file format. The default is the same as input filename.

Outputs:
• The LAMMPS data file ([output].data.in)
• The PSF file ([output].psf)'''

    ##########################################
    ##########################################

    sysTop = _cTOP(top)
    sysTop.SetBondedIndex()
    nBt = len(sysTop.BONDtypeSet)
    nAt = len(sysTop.ANGLEtypeSet)
    nDt = len(sysTop.DIHEDRALtypeSet)
    nB = nA = nD = 0
    for i in range(nBt):
        nB += len(sysTop.BONDtypeSetIdx[i])
    for i in range(nAt):
        nA += len(sysTop.ANGLEtypeSetIdx[i])
    for i in range(nDt):
        nD += len(sysTop.DIHEDRALtypeSetIdx[i])
    allMasses = _np.concatenate(sysTop.BEADms[:])
    allLmpTypes = _np.concatenate(sysTop.LMPtype[:])
    lmpTypes = _np.unique(allLmpTypes)
    lmpMass = _np.zeros(len(lmpTypes), dtype=float)
    for i in range(len(lmpTypes)):
        for j in range(len(allMasses)):
            if lmpTypes[i] == allLmpTypes[j]:
                lmpMass[i] = allMasses[j]
                break
    nBeadTypes = len(lmpTypes)
    nBeads = int(_np.sum(sysTop.nMOL * sysTop.nBEAD))
    lmpTypesSeq = _np.zeros(nBeads, dtype=int)
    BeadchSeq = _np.zeros(nBeads, dtype=float)
    BeadmsSeq = _np.zeros(nBeads, dtype=float)
    BeadTypeSeq = _np.zeros(nBeads, dtype='U5')
    BeadNameSeq = _np.zeros(nBeads, dtype='U5')
    MolNameSeq = _np.zeros(nBeads, dtype='U5')
    n = 0
    for mol in range(len(sysTop.MOL)):
        for i in range(sysTop.nMOL[mol]):
            lmpTypesSeq[n: n + sysTop.nBEAD[mol]] = sysTop.LMPtype[mol]
            BeadchSeq[n: n + sysTop.nBEAD[mol]] = sysTop.BEADch[mol]
            BeadmsSeq[n: n + sysTop.nBEAD[mol]] = sysTop.BEADms[mol]
            BeadTypeSeq[n: n + sysTop.nBEAD[mol]] = sysTop.BEADtype[mol]
            BeadNameSeq[n: n + sysTop.nBEAD[mol]] = sysTop.BEADname[mol]
            MolNameSeq[n: n + sysTop.nBEAD[mol]] = sysTop.MOL[mol]
            n += sysTop.nBEAD[mol]

    sysConf = _cTRAJ(inConf, 'r')
    sysConf.read(_np.sum(sysTop.nMOL * sysTop.nBEAD))
    sysConf.close()
    tric = True
    if sysConf.boxCryst[3] == sysConf.boxCryst[4] == sysConf.boxCryst[5] == 90:
        tric = False
    lx = sysConf.boxMat[0]
    ly = sysConf.boxMat[1]
    lz = sysConf.boxMat[2]
    xy = sysConf.boxMat[3]
    xz = sysConf.boxMat[4]
    yz = sysConf.boxMat[5]
    if sysConf.Type == 'LMP':
        if sysConf.xu is not None:
            X = sysConf.xu
            Y = sysConf.yu
            Z = sysConf.zu
        elif (sysConf.xs is not None) and (sysConf.ix is not None):
            sysConf.unwrapLMPsnap()
            X = sysConf.xu
            Y = sysConf.yu
            Z = sysConf.zu
        elif sysConf.x is not None:
            X = sysConf.x
            Y = sysConf.y
            Z = sysConf.z
        elif sysConf.xs is not None:
            sysConf.unscaleSnap()
            X = sysConf.x
            Y = sysConf.y
            Z = sysConf.z
        else:
            raise _scg4pyError('Error in reading LAMMPS trajectory.')
    else:
        X = sysConf.x
        Y = sysConf.y
        Z = sysConf.z

    if output == None:
        output = _Rstrip(inConf, ['.pdb', '.gro', '.xtc', '.dcd', '.lammpstrj'])
    else:
        output = _Rstrip(output, '.data.in')
    print('\noutput files:')
    with open(output  + '.data.in', 'w') as dataFile:
        print ('   ' + output + '.data.in')
        dataFile.write('LAMMPS data file generated by scg4py python module\n\n')
        dataFile.write('{0:12d}  atoms\n{1:12d}  bonds\n{2:12d}  angles\n{3:12d}  dihedrals\n{4:12d}  impropers\n\n'.format(
            nBeads, nB, nA, nD, 0))
        dataFile.write('{0:12d}  atom types\n{1:12d}  bond types\n'.format(nBeadTypes, nBt))
        dataFile.write('{0:12d}  angle types\n{1:12d}  dihedral types\n{2:12d}  improper types\n\n'.format(nAt, nDt, 0))
        dataFile.write('{0:.8e} {1:.8e} xlo xhi\n'.format(0.0, lx))
        dataFile.write('{0:.8e} {1:.8e} ylo yhi\n'.format(0.0, ly))
        dataFile.write('{0:.8e} {1:.8e} zlo zhi\n'.format(0.0, lz))
        if tric:
            dataFile.write('{0:.8e} {1:.8e} {2:.8e} xy xz yz\n'.format(xy, xz, yz))
        dataFile.write('\nMasses\n\n')
        for i in range(nBeadTypes):
            dataFile.write('{0:7d}  {1:<10.4f}\n'.format(lmpTypes[i], lmpMass[i]))
        dataFile.write('\nAtoms\n\n')
        for i in range(nBeads):
            dataFile.write('{0:7d} {1:7d} {2:5d} {3:.3f} {4:.8e} {5:.8e} {6:.8e}\n'.format(i + 1, sysTop.MOLnum[i] + 1,
                            lmpTypesSeq[i], BeadchSeq[i], X[i], Y[i], Z[i]))
        if nB != 0:
            dataFile.write('\nBonds\n\n')
            n = 1
            for i in range(nBt):
                for j in range(len(sysTop.BONDtypeSetIdx[i])):
                    dataFile.write('{0:6d} {1:4d} {2:7d} {3:7d}\n'.format(n, sysTop.BONDtypeSet[i] + 1,
                                    sysTop.BONDtypeSetIdx[i][j, 0] + 1, sysTop.BONDtypeSetIdx[i][j, 1] + 1))
                    n += 1
        if nA != 0:
            dataFile.write('\nAngles\n\n')
            n = 1
            for i in range(nAt):
                for j in range(len(sysTop.ANGLEtypeSetIdx[i])):
                    dataFile.write('{0:6d} {1:4d} {2:7d} {3:7d} {4:7d}\n'.format(n, sysTop.ANGLEtypeSet[i] + 1,
                                                                                 sysTop.ANGLEtypeSetIdx[i][j, 0] + 1,
                                                                                 sysTop.ANGLEtypeSetIdx[i][j, 1] + 1,
                                                                                 sysTop.ANGLEtypeSetIdx[i][j, 2] + 1))
                    n += 1
        if nD != 0:
            dataFile.write('\nDihedrals\n\n')
            n = 1
            for i in range(nDt):
                for j in range(len(sysTop.DIHEDRALtypeSetIdx[i])):
                    dataFile.write('{0:6d} {1:4d} {2:7d} {3:7d} {4:7d} {5:7d}\n'.format(n, sysTop.DIHEDRALtypeSet[i] + 1,
                                                                                sysTop.DIHEDRALtypeSetIdx[i][j, 0] + 1,
                                                                                sysTop.DIHEDRALtypeSetIdx[i][j, 1] + 1,
                                                                                sysTop.DIHEDRALtypeSetIdx[i][j, 2] + 1,
                                                                                sysTop.DIHEDRALtypeSetIdx[i][j, 3] + 1))
                    n += 1

    with open(output + '.psf', 'w') as psfFile:
        print ('   ' + output + '.psf')
        psfFile.write('PSF file generated by scg4py python module\n\n{0:8d} !NTITLE\n\n'.format(1))
        psfFile.write('{0:8d} !NATOM\n'.format(nBeads))
        for i in range(nBeads):
            psfFile.write('{0:8d} {1:<4s} {2:<4d} {3:<4s} {4:<4s} {5:<4s} {6:10.6f} {7:13.4f} {8:11d}\n'.format(i + 1,
                'U', sysTop.MOLnum[i] + 1, MolNameSeq[i], BeadNameSeq[i], BeadTypeSeq[i], BeadchSeq[i], BeadmsSeq[i], 0))
        psfFile.write('\n{0:8d} !NBOND: bonds\n'.format(nB))
        flag = 0
        for i in range(nBt):
            for j in range(len(sysTop.BONDtypeSetIdx[i])):
                psfFile.write('{0:8d}{1:8d}'.format(sysTop.BONDtypeSetIdx[i][j, 0] + 1,
                                                    sysTop.BONDtypeSetIdx[i][j, 1] + 1))
                flag += 1
                if flag % 4 == 0:
                    psfFile.write('\n')
        flag = 0
        psfFile.write('\n\n{0:8d} !NTHETA: angles\n'.format(nA))
        for i in range(nAt):
            for j in range(len(sysTop.ANGLEtypeSetIdx[i])):
                psfFile.write('{0:8d}{1:8d}{2:8d}'.format(sysTop.ANGLEtypeSetIdx[i][j, 0] + 1,
                                                          sysTop.ANGLEtypeSetIdx[i][j, 1] + 1,
                                                          sysTop.ANGLEtypeSetIdx[i][j, 2] + 1))
                flag += 1
                if flag % 3 == 0:
                    psfFile.write('\n')
        flag = 0
        psfFile.write('\n\n{0:8d} !NPHI: dihedrals\n'.format(nD))
        for i in range(nDt):
            for j in range(len(sysTop.DIHEDRALtypeSetIdx[i])):
                psfFile.write('{0:8d}{1:8d}{2:8d}{3:8d}'.format(sysTop.DIHEDRALtypeSetIdx[i][j, 0] + 1,
                                                            sysTop.DIHEDRALtypeSetIdx[i][j, 1] + 1,
                                                            sysTop.DIHEDRALtypeSetIdx[i][j, 2] + 1,
                                                            sysTop.DIHEDRALtypeSetIdx[i][j, 3] + 1))
                flag += 1
                if flag % 3 == 0:
                    psfFile.write('\n')
        psfFile.write('\n\n%8d !NIMPHI: impropers\n\n%8d !NCRTERM: cross-terms' % (0, 0))

#####################################################################
#####################################################################
#####################################################################

def genLMPscript(pot, top, sysName, cutoff, cutoffSkin=2, maxNB=100, maxB=100, maxA=10,
                  binNB=0.01, binAD=0.05, script4inverse=True, gpu=False):
    '''This program generates a typical script file for simulating on NVT mode with LAMMPS software. Also, it produces tabulated potential files by interpolation and extrapolation of the input potential file. Preparation of the potential tables for LAMMPS software was done as follows: each table is smoothed through radial basis function interpolation. Torsion angle potentials are interpolated periodically while potential tables of bond, angle, and left-hand side of the non-bonded interactions are extrapolated by a quadratic function. The right-hand side of the non-bonded potentials are extrapolated by polynomial functions so that the potential and force at the cutoff radius became zero.

Parameters:
• pot: Input potential file.
• top: CG topology file.
• sysName: The system name used as the base name of the output files.
• cutoff : The non-bonded interaction cutoff distance in Angstrom.
• cutoffSkin: A distance before the cutoff radius used in the extrapolation of the right-hand side of the non-bonded potentials. The default is 2 Angstrom.
• maxNB: The maximum force at distance 0 Angstrom used in the extrapolation of the left-hand side of the non-bonded potential tables. The default is 100 Kcal/mole-Angstrom.
• maxB: The maximum force at distance 0 Angstrom used in the extrapolation of the bond potential tables. The default is 100 Kcal/mole-Angstrom.
• maxA: The maximum force at angle 0 degrees used in the extrapolation of the angle potential tables. The default is 10 Kcal/mole.
• binNB: The bin width used in the interpolation and extrapolation of the bond and non-bonded potential tables. The default is 0.01 Angstrom.
• binAD: The bin width used in the interpolation and extrapolation of the angle and dihedral tables. The default is 0.05 degrees.
• script4inverse: If true, the script file will be compatible with the ’scg4py.runLMP’ and ’scg4py.RefinePot’ programs. Otherwise, the script file can be used in the bash terminal.
• gpu: If true, all the commands required to simulate on GPUs are written in the script file. Be careful that you must compile the LAMMPS software with the GPU package to use this command.

Outputs:
• The LAMMPS script file ([sysName].lmp.in)
• The non-bonded tabulated potential file ([sysName].tab NB)
• The bond tabulated potential file ([sysName].tab B)
• The angle tabulated potential file ([sysName].tab A)
• The dihedral tabulated potential file ([sysName].tab D)'''

    ##########################################
    ##########################################

    LMPscript = sysName + '.lmp.in'
    # NB_tabName = sysName + '.tab_NB'
    # B_tabName = sysName + '.tab_B'
    # A_tabName = sysName + '.tab_A'
    # D_tabName = sysName + '.tab_D'
    maxNB = _np.abs(maxNB)
    maxB = _np.abs(maxB)
    maxA = _np.abs(maxA)
    
    sysTop = _cTOP(top)
    BEADtype = _np.concatenate(sysTop.BEADtype[:])
    BEADtypeSet = _np.zeros(1, dtype='<U5')
    for i in range(len(BEADtype)):
        if BEADtype[i] not in BEADtypeSet:
            BEADtypeSet = _np.append(BEADtypeSet, BEADtype[i])
    BEADtypeSet = BEADtypeSet[1:]
    BEADtypeSetLMPid = _np.zeros(len(BEADtypeSet), dtype=object)
    nBTs = len(BEADtypeSet)
    for i in range(nBTs):
        id = []
        for mol in range(len(sysTop.MOL)):
            tempId = sysTop.BEADtype[mol] == BEADtypeSet[i]
            id = _np.append(id, sysTop.LMPtype[mol][tempId])
        BEADtypeSetLMPid[i] = _np.unique(id)
    nI = int(0.5 * nBTs * (nBTs + 1))
    NonBONDEDtypeSet = _np.zeros([nI, 2], dtype='U5')
    NonBONDEDtypeSetLMPid = _np.zeros(nI, dtype=object)
    n = 0
    for i in range(nBTs):
        idA = BEADtypeSetLMPid[i]
        for j in range(i, nBTs):
            NonBONDEDtypeSet[n, 0] = BEADtypeSet[i]
            NonBONDEDtypeSet[n, 1] = BEADtypeSet[j]
            idB = BEADtypeSetLMPid[j]
            if j != i:
                pairId = _np.zeros([len(idA) * len(idB), 2], dtype=int)
                m = 0
                for k in range(len(idA)):
                    for l in range(len(idB)):
                        pairId[m, 0] = idA[k]
                        pairId[m, 1] = idB[l]
                        m += 1
            else:
                nR = int((len(idA) ** 2 +  len(idA)) / 2)
                pairId = _np.zeros([nR, 2], dtype=int)
                m = 0
                for k in range(len(idA)):
                    for l in range(k, len(idA)):
                        pairId[m, 0] = idA[k]
                        pairId[m, 1] = idA[l]
                        m += 1
            NonBONDEDtypeSetLMPid[n] = pairId
            n += 1

    BONDtype = _np.concatenate(sysTop.BONDtype[:])
    BONDtypeName = _np.concatenate(sysTop.BONDtypeName[:])
    BONDtypeSet = _np.unique(BONDtype)
    BONDtypeSetName = _np.zeros([len(BONDtypeSet), 2], dtype='U5')
    for i in range(len(BONDtypeSet)):
        for j in range(len(BONDtype)):
            if BONDtypeSet[i] == BONDtype[j]:
                BONDtypeSetName[i] = BONDtypeName[j]
                break

    ANGLEtype = _np.concatenate(sysTop.ANGLEtype[:])
    ANGLEtypeName = _np.concatenate(sysTop.ANGLEtypeName[:])
    ANGLEtypeSet = _np.unique(ANGLEtype)
    ANGLEtypeSetName = _np.zeros([len(ANGLEtypeSet), 3], dtype='U5')
    for i in range(len(ANGLEtypeSet)):
        for j in range(len(ANGLEtype)):
            if ANGLEtypeSet[i] == ANGLEtype[j]:
                ANGLEtypeSetName[i] = ANGLEtypeName[j]
                break

    DIHEDRALtype = _np.concatenate(sysTop.DIHEDRALtype[:])
    DIHEDRALtypeName = _np.concatenate(sysTop.DIHEDRALtypeName[:])
    DIHEDRALtypeSet = _np.unique(DIHEDRALtype)
    DIHEDRALtypeSetName = _np.zeros([len(DIHEDRALtypeSet), 4], dtype='U5')
    for i in range(len(DIHEDRALtypeSet)):
        for j in range(len(DIHEDRALtype)):
            if DIHEDRALtypeSet[i] == DIHEDRALtype[j]:
                DIHEDRALtypeSetName[i] = DIHEDRALtypeName[j]
                break

    sysPot = _cTAB(pot, 'r')
    if not (_np.all(BONDtypeSetName == sysPot.BondName) and _np.all(BONDtypeSet == sysPot.BondType)):
        mess = 'Bond terms in the "{0:s}" file are not complible with the "{1:s}" file'.format(pot, top)
        raise _scg4pyError(mess)
    if not (_np.all(ANGLEtypeSetName == sysPot.AngleName) and _np.all(ANGLEtypeSet == sysPot.AngleType)):
        mess = 'Angle terms in the "{0:s}" file are not complible with the "{1:s}" file'.format(pot, top)
        raise _scg4pyError(mess)
    if not (_np.all(DIHEDRALtypeSetName == sysPot.DihedralName) and _np.all(DIHEDRALtypeSet == sysPot.DihedralType)):
        mess = 'Dihedral terms in the "{0:s}" file are not complible with the "{1:s}" file'.format(pot, top)
        raise _scg4pyError(mess)
    if not _np.all(NonBONDEDtypeSet == sysPot.NonBondType):
        mess = 'Non-Bonded terms in the "{0:s}" file are not complible with the "{1:s}" file'.format(pot, top)
        raise _scg4pyError(mess)

    nPairType = len(NonBONDEDtypeSet)
    nBondType = len(BONDtypeSet)
    nAngleType = len(ANGLEtypeSet)
    nDihedralType = len(DIHEDRALtypeSet)

    NBtab = _np.zeros([nPairType, 3], dtype=object) # x, pot, force
    Btab = _np.zeros([nBondType, 3], dtype=object) # x, pot, force
    Atab = _np.zeros([nAngleType, 3], dtype=object) # x, pot, force
    Dtab = _np.zeros([nDihedralType, 3], dtype=object) # x, pot, force

    minPot = _np.zeros(nBondType, dtype=float)
    maxPot = _np.zeros(nBondType, dtype=float)
    minForce = _np.zeros(nBondType, dtype=float)
    maxForce = _np.zeros(nBondType, dtype=float)
    if nBondType > 0:
        print ('BOND tables: pot= (min , max) [Kcal/mol] , force= (min , max) [Kcal/mol-Angstrom]:')
    for i in range(nBondType):
        x = sysPot.BondX[i]
        pot = sysPot.BondY[i]
        dpot = -1 * _np.diff(pot) / _np.diff(x)
        minP, maxP = _np.min(pot), _np.max(pot)
        minF, maxF = _np.min(dpot), _np.max(dpot)
        minPot[i], maxPot[i] = minP, maxP
        minForce[i], maxForce[i] = minF, maxF
        print('  {0:^5d} : pot = ({1:^9.4f} , {2:^9.4f}) , force = ({3:^9.4f} , {4:^9.4f})'.format(
            BONDtypeSet[i], minP, maxP, minF, maxF))
        X, Pot, Force = sysPot.extrpPot_Bond(i, binNB, maxB, cutoff)
        Btab[i, 0] = X
        Btab[i, 1] = Pot
        Btab[i, 2] = Force
    if nBondType > 0:
        print('  -----------------------------------------------------------------------')
        print('  Total : pot = ({0:^9.4} , {1:^9.4f}) , force = ({2:^9.4f} , {3:^9.4f})'.format(
            _np.min(minPot), _np.max(maxPot), _np.min(minForce), _np.max(maxForce)))

    minPot = _np.zeros(nAngleType, dtype=float)
    maxPot = _np.zeros(nAngleType, dtype=float)
    minForce = _np.zeros(nAngleType, dtype=float)
    maxForce = _np.zeros(nAngleType, dtype=float)
    if nAngleType > 0:
        print('\nAngle tables: pot= (min , max) [Kcal/mol] , force= (min , max) [Kcal/mol]:')
    for i in range(nAngleType):
        x = sysPot.AngleX[i]
        pot = sysPot.AngleY[i]
        dpot = -1 * _np.diff(pot) / _np.diff(x)
        minP, maxP = _np.min(pot), _np.max(pot)
        minF, maxF = _np.min(dpot), _np.max(dpot)
        minPot[i], maxPot[i] = minP, maxP
        minForce[i], maxForce[i] = minF, maxF
        print('  {0:^5d} : pot = ({1:^9.4f} , {2:^9.4f}) , force = ({3:^9.4f} , {4:^9.4f})'.format(
            ANGLEtypeSet[i], minP, maxP, minF, maxF))
        X, Pot, Force = sysPot.extrpPot_Angle(i, binAD, maxA)
        Atab[i, 0] = X
        Atab[i, 1] = Pot
        Atab[i, 2] = Force
    if nAngleType > 0:
        print('  -----------------------------------------------------------------------')
        print('  Total : pot = ({0:^9.4} , {1:^9.4f}) , force = ({2:^9.4f} , {3:^9.4f})'.format(
            _np.min(minPot), _np.max(maxPot), _np.min(minForce), _np.max(maxForce)))

    minPot = _np.zeros(nDihedralType, dtype=float)
    maxPot = _np.zeros(nDihedralType, dtype=float)
    minForce = _np.zeros(nDihedralType, dtype=float)
    maxForce = _np.zeros(nDihedralType, dtype=float)
    if nDihedralType > 0:
        print ('\nDIHEDRAL tables: pot= (min , max) [Kcal/mol] , force= (min , max) [Kcal/mol]:')
    for i in range(nDihedralType):
        x = sysPot.DihedralX[i]
        pot = sysPot.DihedralY[i]
        dpot = -1 * _np.diff(pot) / _np.diff(x)
        minP, maxP = _np.min(pot), _np.max(pot)
        minF, maxF = _np.min(dpot), _np.max(dpot)
        minPot[i], maxPot[i] = minP, maxP
        minForce[i], maxForce[i] = minF, maxF
        print('  {0:^5d} : pot = ({1:^9.4f} , {2:^9.4f}) , force = ({3:^9.4f} , {4:^9.4f})'.format(
            DIHEDRALtypeSet[i], minP, maxP, minF, maxF))
        X, Pot, Force = sysPot.extrpPot_Dihedral(i, binAD)
        Dtab[i, 0] = X
        Dtab[i, 1] = Pot
        Dtab[i, 2] = Force
    if nDihedralType > 0:
        print('  -----------------------------------------------------------------------')
        print('  Total : pot = ({0:^9.4} , {1:^9.4f}) , force = ({2:^9.4f} , {3:^9.4f})'.format(
            _np.min(minPot), _np.max(maxPot), _np.min(minForce), _np.max(maxForce)))

    minPot = _np.zeros(nPairType, dtype=float)
    maxPot = _np.zeros(nPairType, dtype=float)
    minForce = _np.zeros(nPairType, dtype=float)
    maxForce = _np.zeros(nPairType, dtype=float)
    print ('\nNON-BONDED tables: pot= (min , max) [Kcal/mol] , force= (min , max) [Kcal/mol-Angstrom]:')
    for i in range(nPairType):
        x = sysPot.NonBondX[i]
        pot = sysPot.NonBondY[i]
        dpot = -1 * _np.diff(pot) / _np.diff(x)
        minP, maxP = _np.min(pot), _np.max(pot)
        minF, maxF = _np.min(dpot), _np.max(dpot)
        minPot[i], maxPot[i] = minP, maxP
        minForce[i], maxForce[i] = minF, maxF
        print('  {0:>5s}-{1:<5s}: pot = ({2:^9.4f} , {3:^9.4f}) , force = ({4:^9.4f} , {5:^9.4f})'.format(
            NonBONDEDtypeSet[i, 0], NonBONDEDtypeSet[i, 1], minP, maxP, minF, maxF))
        X, Pot, Force = sysPot.extrpPot_NonBonded(i, binNB, maxNB, cutoff, cutoffSkin)
        NBtab[i, 0] = X
        NBtab[i, 1] = Pot
        NBtab[i, 2] = Force
    if nPairType > 0:
        print('     -------------------------------------------------------------------------')
        print('     Total   : pot = ({0:^9.4} , {1:^9.4f}) , force = ({2:^9.4f} , {3:^9.4f})'.format(
            _np.min(minPot), _np.max(maxPot), _np.min(minForce), _np.max(maxForce)))

    # _np.save(sysName+'.FF', [Btab, Atab, Dtab, NBtab])
    print('\noutput files:')
    _writeLMPtab(sysName, Btab, BONDtypeSet, Atab, ANGLEtypeSet, Dtab, DIHEDRALtypeSet, NBtab, NonBONDEDtypeSet)
    nBeadCumsum = _np.cumsum(sysTop.nBEAD * sysTop.nMOL)
    nBeadCumsum = _np.append([0], nBeadCumsum)
    BEADch = _np.concatenate(sysTop.BEADch[:])
    with open(LMPscript, 'w') as lmpFile:
        print (LMPscript)
        lmpFile.write('# -------------------- Variables ---------------------\n\n')
        lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
            'variable', 'dt', 'equal', '10', '# timestep [fs]'))
        lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
            'variable', 'run', 'equal', '500', '# NVT md  [ns]'))
        lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
            'variable', 'nstlist', 'string', '5', '# number of steps neighbour list updating'))
        lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
            'variable', 'bin', 'string', '5', '# neighbor searching bin'))
        lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
            'variable', 'thermo', 'equal', '50', '# output thermodynamics [ps]'))
        lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
            'variable', 'traj', 'equal', '50', '# output trajectory [ps]'))
        lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
            'variable', 'Tstart', 'string', '300', '# starting temperature [K]'))
        lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
            'variable', 'Tstop', 'string', '300', '# ending temperature [K]'))
        lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
            'variable', 'LDdamp', 'string', '500', '# Langevin Dynamics damping parameter [fs]'))
        if not script4inverse:
            lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:<15d} {4:s}\n'.format(
                'variable', 'seed', 'equal', _np.random.randint(10,9999,1)[0], '# random seed'))
            lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
                'variable', 'system', 'string', sysName, '# system name'))
            lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
                'variable', 'restart', 'string', 'no', '# if yes: input data will be RestartFile variable '))
            lmpFile.write(' {0:11s} {1:14s} {2:9s} {3:15s} {4:s}\n'.format(
                'variable', 'RestartFile', 'string', sysName+'.restart1', ' # restart file name'))

        lmpFile.write('\n# ------------------ Initialization ------------------\n\n')
        if gpu:
            lmpFile.write(' package gpu 2 neigh no split 1 tpa 32\n')
            tabStyle = 'table/gpu'
            coulStyle = 'coul/long/gpu'
        else:
            tabStyle = 'table'
            coulStyle = 'coul/long'
        lmpFile.write(' {0:17s}{1:s}\n'.format('units', 'real'))
        lmpFile.write(' {0:17s}{1:s}\n'.format('atom_style', 'full'))
        if _np.any(BEADch != 0.0):
            lmpFile.write(' {0:17s}{1:s}   {2:s} {3:s} {4:d}  {5:s} {6:f}\n'.format(
                'pair_style', 'hybrid/overlay', tabStyle, 'linear', len(NBtab[0, 0]) - 1, coulStyle, float(cutoff)))
        else:
            lmpFile.write(' {0:17s}{1:s} {2:s} {3:d}\n'.format('pair_style', tabStyle, 'linear', len(NBtab[0, 0]) - 1))
        if nBondType > 0:
            lmpFile.write(' {0:17s}{1:s} {2:s} {3:d}\n'.format('bond_style', 'table', 'linear', len(Btab[0, 0]) - 1))
        if nAngleType > 0:
            lmpFile.write(' {0:17s}{1:s} {2:s} {3:d}\n'.format('angle_style', 'table', 'linear', len(Atab[0, 0])))
        if nDihedralType > 0:
            lmpFile.write(' {0:17s}{1:s} {2:s} {3:d}\n'.format('dihedral_style', 'table', 'linear', len(Dtab[0, 0]) - 1))
        lmpFile.write('\n# ----------------- Atoms definition -----------------\n\n')
        lmpFile.write(' if "${restart} == yes" then "read_restart ${RestartFile}" else "read_data ${system}.data.in"\n')
        if _np.any(BEADch != 0.0):
            lmpFile.write(' {0:17s}{1:s} {2:s}\n'.format('kspace_style', 'pppm/cg', '1.0e-5 1.0e-3'))
            lmpFile.write(' {0:17s}{1:s}\n'.format('dielectric', '78'))
        lmpFile.write('\n# --------------------- Settings ---------------------\n\n')
        for i in range(len(NonBONDEDtypeSet)):
            tA = NonBONDEDtypeSet[i, 0]
            tB = NonBONDEDtypeSet[i, 1]
            pairName = 'Non-Bonded_{0:s}-{1:s}'.format(tA, tB)
            for j in range(len(NonBONDEDtypeSetLMPid[i])):
                id1 = NonBONDEDtypeSetLMPid[i][j, 0]
                id2 = NonBONDEDtypeSetLMPid[i][j, 1]
                if _np.any(BEADch != 0.0):
                    lmpFile.write(' {0:s}{1:5d}{2:5d}  {3:s}  {4:s}  {5:25s} {6:.3f}\n'.format(
                        'pair_coeff', id1, id2, tabStyle, '${system}.tab_NB', pairName, float(cutoff)))
                else:
                    lmpFile.write(' {0:s}{1:5d}{2:5d}  {3:s}  {4:25s} {5:.3f}\n'.format(
                        'pair_coeff', id1, id2, '${system}.tab_NB', pairName, float(cutoff)))
        if _np.any(BEADch != 0.0):
            lmpFile.write(' pair_coeff    *    *  {0:s}\n'.format(coulStyle))
        if nBondType > 0:
            lmpFile.write('\n')
            for i in range(nBondType):
                lmpFile.write(' {0:s}{1:5d}   {2:s}  {3:s}{4:d}\n'.format(
                    'bond_coeff', BONDtypeSet[i] + 1, '${system}.tab_B', 'Bond_', BONDtypeSet[i] + 1))
        if nAngleType > 0:
            lmpFile.write('\n')
            for i in range(nAngleType):
                lmpFile.write(' {0:s}{1:5d}   {2:s}  {3:s}{4:d}\n'.format(
                    'angle_coeff', ANGLEtypeSet[i] + 1, '${system}.tab_A', 'Angle_', ANGLEtypeSet[i] + 1))
        if nDihedralType > 0:
            lmpFile.write('\n')
            for i in range(nDihedralType):
                lmpFile.write(' {0:s}{1:5d}   {2:s}  {3:s}{4:d}\n'.format(
                    'dihedral_coeff', DIHEDRALtypeSet[i] + 1, '${system}.tab_D', 'Dihedral_', DIHEDRALtypeSet[i] + 1))
        if _np.all(BEADch == 0):
            keyword = 'lj'
        else:
            keyword = 'lj/coul'
        if sysTop.EXCL == 1:
            lmpFile.write('\n special_bonds {} 0.0 1.0 1.0'.format(keyword))
        elif sysTop.EXCL == 2:
            lmpFile.write('\n special_bonds {} 0.0 0.0 1.0'.format(keyword))
        elif sysTop.EXCL == 3:
            lmpFile.write('\n special_bonds {} 0.0 0.0 0.0'.format(keyword))
        else:
            pass
        lmpFile.write('\n# ------------------- Minimization -------------------\n\n')
        lmpFile.write(''' if "${restart} == yes" then "write_data ${system}.remap"
 neighbor ${bin} bin
 neigh_modify every 1 delay 0 check yes
 thermo_style custom step epair emol fmax
 thermo 200
 min_style cg
 if "${restart} == no" then "minimize 0.0 0.0 10000 1000000" "reset_timestep 0"\n''')
        lmpFile.write('\n# --------------------- NVT RUN ---------------------\n\n')
        for i in range(len(sysTop.MOL)):
            lmpFile.write(' group {0:s} id {1:d}:{2:d}\n'.format(sysTop.MOL[i], nBeadCumsum[i] + 1, nBeadCumsum[i+1]))
        lmpFile.write(''' timestep $(v_dt)
 neigh_modify every ${nstlist} delay 0 check yes
 if "${restart} == no" then "velocity all create ${Tstart} $(v_seed) dist gaussian"
 fix B all balance $(floor(500000 / v_dt)) 1.1 shift xyz 20 1.1\n''')
        if len(sysTop.MOL) == 1:
            lmpFile.write(' fix L all langevin ${Tstart} ${Tstop} ${LDdamp} $(v_seed) zero yes\n')
        else:
            for i in range(len(sysTop.MOL)):
                if i == 0:
                    lmpFile.write(' fix L%d %s langevin ${Tstart} ${Tstop} ${LDdamp} $(v_seed) zero yes\n' % (
                        i + 1, sysTop.MOL[i]))
                else:
                    lmpFile.write(' fix L%d %s langevin ${Tstart} ${Tstop} ${LDdamp} $(v_seed + %d) zero yes\n' % (
                        i + 1, sysTop.MOL[i], i))
        lmpFile.write(''' fix N all nve
 fix M all momentum 10 linear 1 1 1
 fix R all recenter INIT INIT INIT units box 
 thermo_style multi 
 thermo $(floor(v_thermo * 1e3 / v_dt))
 write_restart ${system}.restart1
 restart $(floor(500000 / v_dt)) ${system}.restart1 ${system}.restart2
 dump D all atom/mpiio $(floor(v_traj * 1e3 / v_dt)) ${system}.mpiio.lammpstrj
 dump_modify D sort id
 run $(floor(v_run * 1e6 / v_dt)) upto 
 write_data ${system}.data.out ''')

#####################################################################
#####################################################################
#####################################################################

def runLMP(sysName, LMP, MPI=None, nProc=None, restart=False, maxRestarting=10):
    '''This program uses outputs of ’scg4py.genLMPscript’ and ’scg4py.genLMPdata’ to run the CG simulation with LAMMPS software.

Parameters:
• sysName: The system name used as the base name of the input files.
• LMP: The path of the LAMMPS executable.
• MPI: The path of the mpirun executable. By default, the MPI executable is not used.
• nProc: The number of parallel processes. The default is None.
• restart: If True, Simulation continues from the previous restart point. The default is False.
• maxRestarting: The maximum number of restarting if the simulation was interrupted. If the simulation is interrupted frequently, this indicates bad physics, e.g. too large a timestep, etc. The default is 10.'''

    ##########################################
    ##########################################

    def searchFile(sysName, ext):
        dirL = _glob.glob(sysName + '.*' + ext)
        suffix = []
        for line in dirL:
            temp = _Rstrip(line, '.' + ext).split('.')
            if len(temp) == 2:
                suffix.append(int(temp[1]))
        suffix.sort()
        nameList = [sysName + '.' + str(i) for i in suffix]
        if len(nameList) > 0:
            maxSuffix = max(suffix)
        else:
            maxSuffix = -1
        return nameList, maxSuffix

    script = sysName + '.lmp.in'
    resList = [sysName + '.restart1', sysName + '.restart2']
    remap = sysName + '.remap'
    if _os.path.isfile(remap):
        _os.remove(remap)
    fid = open(script)
    scList = fid.readlines()
    fid.close()
    dumpLine = ''
    for line in scList:
        lsp = line.split('#')[0]
        if 'dump' in lsp and 'dump_' not in lsp:
            dumpLine = lsp
            break
    if '.mpiio.lammpstrj' in dumpLine:
        trajExt = 'mpiio.lammpstrj'
    elif '.lammpstrj' in dumpLine:
        trajExt = 'lammpstrj'
    elif '.xtc' in dumpLine:
        trajExt = 'xtc'
    elif '.dcd' in dumpLine:
        trajExt = 'dcd'
    else:
        mess = 'type of trajectory should be one of ".mpiio.lammpstrj", ".lammpstrj", ".xtc" or ".dcd".'
        raise _scg4pyError(mess)

    trajNum = 0
    logNum = 0
    if restart:
        resCheck = [_os.path.isfile(i) for i in resList]
        if not any(resCheck):
            mess = 'no restart file(s).\ntry with "restart=False".'
            raise _scg4pyError(mess)
        dataList, dataNum = searchFile(sysName, 'data.out')
        dataNum += 1
        if _os.path.isfile(sysName + '.data.out'):
            _os.rename(sysName + '.data.out', sysName + '.' + str(dataNum) + '.data.out')
        trajList, trajNum = searchFile(sysName, trajExt)
        trajNum += 1
        if _os.path.isfile(sysName + '.' + trajExt):
            _os.rename(sysName + '.' + trajExt, sysName + '.' + str(trajNum) + '.' + trajExt)
            trajNum += 1
        logList, logNum = searchFile(sysName, 'log')
        logNum += 1
        if _os.path.isfile(sysName + '.log'):
            _os.rename(sysName + '.' + 'log', sysName + '.' + str(logNum) + '.log')
            logNum += 1
    else:
        if _os.path.isfile(sysName + '.data.out'):
            mess = 'simulation has been finished.\nfor new simulation, remove "{}" file and try again.'.format(
                sysName + '.data.out')
            raise _scg4pyError(mess)
        if not _os.path.isfile(sysName + '.data.in'):
            mess = 'no input file: "{}".'.format(sysName + '.data.in')
            raise _scg4pyError(mess)
        for f in _glob.glob(sysName + '.*' + trajExt):
            _os.remove(f)
        for f in _glob.glob(sysName + '.*.log'):
            _os.remove(f)
        for f in resList:
            if _os.path.isfile(f):
                _os.remove(f)
    np = str(nProc)
    seed = _np.random.choice(range(10, maxRestarting * 10000, 20), maxRestarting, replace=False)
    for runNum in range(maxRestarting):
        if not restart:
            print ('   LAMMPS is running')
            runLMPcmd = [LMP, '-i', script, '-v', 'seed', str(seed[runNum]), '-v', 'system', sysName, '-v', 'restart',
                         'no', '-e', 'log', '-l', sysName + '.log', '-nc']
            if MPI is not None and np.isdigit():
                cmd = [MPI, '-np', np]
                cmd.extend(runLMPcmd)
                proc = _subP.Popen(cmd, stdout=_subP.PIPE, stderr=_subP.PIPE)
                stdout, stderr = proc.communicate()
            else:
                proc = _subP.Popen(runLMPcmd, stdout=_subP.PIPE, stderr=_subP.PIPE)
                stdout, stderr = proc.communicate()
        else:
            for i in range(2):
                if _os.path.isfile(resList[i]):
                    print ('   LAMMPS is running: restart ')
                    runLMPcmd_res = [LMP, '-i', script, '-v', 'seed', str(seed[runNum]), '-v', 'system', sysName, '-v',
                                     'restart', 'yes', '-v', 'RestartFile', resList[i], '-e', 'log',
                                     '-l', sysName + '.log', '-nc']
                    if MPI is not None and np.isdigit():
                        cmd = [MPI, '-np', np]
                        cmd.extend(runLMPcmd_res)
                        proc = _subP.Popen(cmd, stdout=_subP.PIPE, stderr=_subP.PIPE)
                        stdout, stderr = proc.communicate()
                    else:
                        proc = _subP.Popen(runLMPcmd_res, stdout=_subP.PIPE, stderr=_subP.PIPE)
                        stdout, stderr = proc.communicate()
                    if i == 1 and not _os.path.isfile(remap):
                        mess = 'error in parsing the restart file(s).\ntry with "restart=False".'
                        raise _scg4pyError(mess)
                    if _os.path.isfile(remap):
                        _os.remove(remap)
                        break
        _time.sleep(0.5)
        if proc.returncode == 0:
            if restart:
                if _os.path.isfile(sysName + '.log'):
                    _os.rename(sysName + '.log', sysName + '.' + str(logNum) + '.log')
                    logNum += 1
                if _os.path.isfile(sysName + '.' + trajExt):
                    _os.rename(sysName + '.' + trajExt, sysName + '.' + str(trajNum) + '.' + trajExt)
                    trajNum += 1
            break
        else:
            if _os.path.isfile(resList[0]):
                if _os.path.isfile(sysName + '.log'):
                    _os.rename(sysName + '.log', sysName + '.' + str(logNum) + '.log')
                    logNum += 1
                if _os.path.isfile(sysName + '.' + trajExt):
                    _os.rename(sysName + '.' + trajExt, sysName + '.' + str(trajNum) + '.' + trajExt)
                    trajNum += 1
                restart = True
            elif runNum == (maxRestarting):
                mess = 'exceeded "maxRestarting"'
                raise _scg4pyError
            else:
                mess = 'Error in LAMMPS running.'
                raise _scg4pyError(mess)

#####################################################################
#####################################################################
#####################################################################

def parseLMPLog(logFile, timeStep, plot=True):
    '''This program parses the LAMMPS log file. Only ’thermo style multi’ of the log file is acceptable.

Parameters:
• logFile: Input LAMMPS log file.
• timeStep: The simulation time step in femtosecond.
• plot: If true, the plot of each parameter over time is saved.

Outputs:
• A figure for the total energy over time([logFile].TotEng.png)
• A figure for the kinetic energy over time([logFile].KinEng.png)
• A figure for the temperature energy over time([logFile].Temp.png)
• A figure for the potential energy over time([logFile].PotEng.png)
• A figure for the bond energy over time([logFile].E bond.png)
• A figure for the angle energy over time([logFile].E angle.png)
• A figure for the dihedral energy over time([logFile].E dihed.png)
• A figure for the improper dihedral energy over time([logFile].E impro.png)
• A figure for the Van der Waals pairwise energy over time([logFile].E vdwl.png)
• A figure for the coulombic pairwise energy over time([logFile].E coul.png)
• A figure for the long-range kspace energy over time([logFile].E long.png)
• A figure for the pressure energy over time([logFile].Press.png)
• A figure for the volume energy over time([logFile].Volume.png)

Return variables:
• Step, CPU, TotEng, KinEng, Temp, PotEng, E bond, E angle, E dihed, E impro, E vdwl, E coul, E long, Press, Volume'''

    ##########################################
    ##########################################

    logFile = str(logFile)
    fid = open(logFile)
    logL = fid.readlines()
    fid.close()
    timeStep = float(timeStep)

    nStepT = 0
    for l in logL:
        if ('Step' in l) and ('CPU' in l) and ('sec' in l):
            nStepT += 1
    Step = _np.zeros(nStepT, dtype=int)
    CPU = _np.zeros(nStepT, dtype=float)
    TotEng = _np.zeros(nStepT, dtype=float)
    KinEng = _np.zeros(nStepT, dtype=float)
    Temp = _np.zeros(nStepT, dtype=float)
    PotEng = _np.zeros(nStepT, dtype=float)
    E_bond = _np.zeros(nStepT, dtype=float)
    E_angle = _np.zeros(nStepT, dtype=float)
    E_dihed = _np.zeros(nStepT, dtype=float)
    E_impro = _np.zeros(nStepT, dtype=float)
    E_vdwl = _np.zeros(nStepT, dtype=float)
    E_coul = _np.zeros(nStepT, dtype=float)
    E_long = _np.zeros(nStepT, dtype=float)
    Press = _np.zeros(nStepT, dtype=float)
    Volume = _np.zeros(nStepT, dtype=float)

    l = 0
    n = 0
    PrevStep = -1
    while l < len(logL):
        line = logL[l]
        if ('Step' in line) and ('CPU' in line) and ('sec' in line):
            step = int(line.split('Step')[1].split('-')[0])
            cpu = float(line.split('CPU')[1].split('=')[1].split('(sec)')[0])
            l += 1
            if l < len(logL):
                line = logL[l]
            else:
                break
            if ('TotEng' in line) and ('KinEng' in line) and ('Temp' in line):
                toteng = float(line.split('=')[1].split('KinEng')[0])
                kineng = float(line.split('=')[2].split('Temp')[0])
                temp = float(line.split('=')[3])
                l += 1
                if l < len(logL):
                    line = logL[l]
                else:
                    break
                if ('PotEng' in line) and ('E_bond' in line) and ('E_angle' in line):
                    poteng = float(line.split('=')[1].split('E_bond')[0])
                    e_bond = float(line.split('=')[2].split('E_angle')[0])
                    e_angle = float(line.split('=')[3])
                    l += 1
                    if l < len(logL):
                        line = logL[l]
                    else:
                        break
                    if ('E_dihed' in line) and ('E_impro' in line) and ('E_vdwl' in line):
                        e_dihed = float(line.split('=')[1].split('E_impro')[0])
                        e_impro = float(line.split('=')[2].split('E_vdwl')[0])
                        e_vdwl = float(line.split('=')[3])
                        l += 1
                        if l < len(logL):
                            line = logL[l]
                        else:
                            break
                        if ('E_coul' in line) and ('E_long' in line) and ('Press' in line):
                            e_coul = float(line.split('=')[1].split('E_long')[0])
                            e_long = float(line.split('=')[2].split('Press')[0])
                            press = float(line.split('=')[3])
                            l += 1
                            if l < len(logL):
                                line = logL[l]
                            else:
                                break
                            if ('Volume' in line):
                                volume = float(line.split('=')[1])
                                if PrevStep < step:
                                    Step[n] = step
                                    CPU[n] = cpu
                                    TotEng[n] = toteng
                                    KinEng[n] = kineng
                                    Temp[n] = temp
                                    PotEng[n] =poteng
                                    E_bond[n] = e_bond
                                    E_angle[n] = e_angle
                                    E_dihed[n] = e_dihed
                                    E_impro[n] = e_impro
                                    E_vdwl[n] = e_vdwl
                                    E_coul[n] = e_coul
                                    E_long[n] = e_long
                                    Press[n] = press
                                    Volume[n] = volume
                                    n += 1
                                    PrevStep = step
        l += 1
    Step = Step[0:n]
    CPU = CPU[0:n]
    TotEng = TotEng[0:n]
    KinEng = KinEng[0:n]
    Temp = Temp[0:n]
    PotEng = PotEng[0:n]
    E_bond = E_bond[0:n]
    E_angle = E_angle[0:n]
    E_dihed = E_dihed[0:n]
    E_impro = E_impro[0:n]
    E_vdwl = E_vdwl[0:n]
    E_coul = E_coul[0:n]
    E_long = E_long[0:n]
    Press = Press[0:n]
    Volume = Volume[0:n]
    E_bonded = E_bond + E_angle + E_dihed + E_impro
    if plot:
        Step = Step * timeStep / 1000.0
        outN = logFile.split('/')[-1].split('.log')[0]
        outLog = outN + '.parse.log'
        fid = open(outLog, 'w')
        fid.write('# Time [ps], TotEng, KinEng, Temp, PotEng, E_bond, E_angle, E_dihed, E_impro, E_vdwl, E_coul, E_long, Press, Volume\n')
        for i in range(len(Step)):
            fid.write('{0:f} {1:f} {2:f} {3:f} {4:f} {5:f} {6:f} {7:f} {8:f} {9:f} {10:f} {11:f} {12:f} {13:f} {14:f}\n'.format(
                Step[i], TotEng[i], KinEng[i], Temp[i], PotEng[i], E_bond[i], E_angle[i],
                E_dihed[i], E_impro[i], E_bonded[i], E_vdwl[i], E_coul[i], E_long[i], Press[i], Volume[i]))
        fid.close()
        param = ['TotEng', 'KinEng', 'Temp', 'PotEng', 'E_bond', 'E_angle', 'E_dihed',
                 'E_impro', 'E_bonded', 'E_vdwl', 'E_coul', 'E_long', 'Press', 'Volume']
        for i in param:
            _plt.figure(figsize=(14, 7))
            eval("_plt.plot(Step[1:]," + i + "[1:],'or',Step[1:]," + i + "[1:])")
            _plt.title(i, fontsize=16)
            _plt.xlabel('Time [ps]', fontsize=14)
            if i == 'Temp':
                _plt.ylabel('Temperatue [kelvin]', fontsize=14)
            elif i == 'Press':
                _plt.ylabel('pressure [atmospheres]', fontsize=14)
            elif i == 'Volume':
                _plt.ylabel('Volume [Angstroms ^ 3]', fontsize=14)
            else:
                _plt.ylabel('Energy [kCal/mol]', fontsize=14)
            _plt.savefig(outN + '.' + i+'.png')
            _plt.clf()
            _plt.close()
    else:
        return Step, CPU, TotEng, KinEng, Temp, PotEng, E_bond, E_angle, E_dihed, E_impro, E_vdwl, E_coul, E_long, Press, Volume

#####################################################################
#####################################################################
#####################################################################

def trajCat(inTrajs, outTraj, logging=True):
    '''This program concatenates input trajectory files in sorted order.

Parameters:
• inTrajs: List of input trajectory files.
• outTraj: Output trajectory file.
• logging: If true, the time and the number of snapshots of input trajectories are displayed on the screen.'''

    ##########################################
    ##########################################

    if isinstance(inTrajs, list):
        trajList = inTrajs
    else:
        trajList = [inTrajs]
    for i in range(len(trajList)):
        if outTraj == trajList[i]:
            mess = 'output traj file must be different from input ones.'
            raise _scg4pyError(mess)
    if outTraj.endswith('.lammpstrj'):
        ch = [trj.endswith('.lammpstrj') for trj in trajList]
        if not all(ch):
            mess = 'incompatible input and output trajectories.'
            raise _scg4pyError(mess)
    elif outTraj.endswith('.gro'):
        ch = [trj.endswith('.gro') for trj in trajList]
        if not all(ch):
            mess = 'incompatible input and output trajectories.'
            raise _scg4pyError(mess)
    elif outTraj.endswith('.pdb'):
        ch = [trj.endswith('.pdb') for trj in trajList]
        if not all(ch):
            mess = 'incompatible input and output trajectories.'
            raise _scg4pyError(mess)
    elif outTraj.endswith('.dcd'):
        ch = [trj.endswith('.dcd') for trj in trajList]
        if not all(ch):
            mess = 'incompatible input and output trajectories.'
            raise _scg4pyError(mess)
    elif outTraj.endswith('.xtc'):
        ch = [trj.endswith('.xtc') for trj in trajList]
        if not all(ch):
            mess = 'incompatible input and output trajectories.'
            raise _scg4pyError(mess)
    if logging:
        print('Input trajecotries:')
        for i in range(len(trajList)):
            print('\t' + trajList[i])
        print('Output trajecotry:')
        print('\t' + outTraj + '\n')
    trajO = _cTRAJ(outTraj, 'w')
    T = S = -1
    n = 1
    nSnap = _np.ones(len(trajList) + 1, dtype=int)
    for i in range(len(trajList)):
        trajI = _cTRAJ(trajList[i], 'r')
        while True:
            trajI.read()
            if trajI.eof:
                trajI.close()
                nSnap[i + 1] = n - nSnap[i - 1]
                if logging:
                    print('{0:s}: {1:d} snaps'.format(trajList[i], nSnap[i + 1]))
                break
            trajO.time = trajI.time
            trajO.timeStep = trajI.timeStep
            trajO.nAtom = trajI.nAtom
            trajO.atId = trajI.atId
            trajO.atName = trajI.atName
            trajO.atType = trajI.atType
            trajO.resId = trajI.resId
            trajO.resName = trajI.resName
            trajO.x = trajI.x
            trajO.y = trajI.y
            trajO.z = trajI.z
            trajO.xs = trajI.xs
            trajO.ys = trajI.ys
            trajO.zs = trajI.zs
            trajO.xu = trajI.xu
            trajO.yu = trajI.yu
            trajO.zu = trajI.zu
            trajO.ix = trajI.ix
            trajO.iy = trajI.iy
            trajO.iz = trajI.iz
            trajO.fx = trajI.fx
            trajO.fy = trajI.fy
            trajO.fz = trajI.fz
            trajO.lmpBox = trajI.lmpBox
            trajO.boxMat = trajI.boxMat
            trajO.boxCryst = trajI.boxCryst
            if trajI.Type == 'DCD' or trajI.Type == 'PDB' or trajI.Type == 'GRO':
                trajO.write(n)
                n += 1
            else:
                if trajI.time is not None and trajI.time > T:
                    trajO.write()
                    T = trajI.time
                    n += 1
                elif trajI.timeStep is not None and trajI.timeStep > S:
                    trajO.write()
                    S = trajI.timeStep
                    n += 1
    trajO.close()

#####################################################################
#####################################################################
#####################################################################

def trajConv(inTraj, top, outTraj=None, begin=1, end=-1, stride=1, rmPBC=True, lastSnap=False):
    '''This program converts the input CG trajectory in ways of cutting the trajectory, removing periodic boundary condition, and changing the file format.

Parameters:
• inTraj: Input trajectory file.
• top: CG topology file.
• outTraj: Output trajectory file.
• begin: The number of the first frame to read from the input trajectory.
• end: The number of the last frame to read from the input trajectory.
• stride: Only write every nr-th frame.
• rmPBC: If True, it removes the periodic boundary condition. The default is True.
• lastSnap: If True, it stores only last snapshot. The default is False.'''

    ##########################################
    ##########################################

    trajI = _cTRAJ(inTraj, 'r')
    if lastSnap:
        suffix = 'lastSnap'
    elif rmPBC:
        suffix = 'noPBC'
    else:
        suffix = 'new'
    if outTraj is None:
        outTraj = _Rstrip(inTraj, ['.lammpstrj', '.pdb', '.gro', '.dcd', '.xtc']) + '.' + suffix
        if trajI.Type == 'LMP':
            outTraj += '.lammpstrj'
        elif trajI.Type == 'GRO':
            outTraj += '.gro'
        elif trajI.Type == 'DCD':
            outTraj += '.dcd'
        elif trajI.Type == 'PDB':
            outTraj += '.pdb'
        elif trajI.Type == 'XTC':
            outTraj += '.xtc'
    if outTraj == inTraj:
        mess = 'output traj file "{0:s}" must be different from input one "{1:s}".'.format(inTraj, outTraj)
        raise _scg4pyError(mess)
    start_time = _time.time()
    sysTop = _cTOP(top)
    sysTop.calcBondsMap()
    nAtom = int(_np.sum(sysTop.nMOL * sysTop.nBEAD))
    AtCSum = _np.append([0], _np.cumsum(sysTop.nBEAD * sysTop.nMOL))
    MOLid = _np.zeros(nAtom, dtype=int)
    MOLname = _np.zeros(nAtom, dtype='<U5')
    BEADname = _np.zeros(nAtom, dtype='<U5')
    LMPtypeId = _np.zeros(nAtom, dtype=int)
    m = 1
    for mol in range(len(sysTop.MOL)):
        MOLname[AtCSum[mol]:AtCSum[mol + 1]] = _np.tile(sysTop.MOL[mol], sysTop.nBEAD[mol] * sysTop.nMOL[mol])
        BEADname[AtCSum[mol]:AtCSum[mol + 1]] = _np.tile(sysTop.BEADname[mol], sysTop.nMOL[mol])
        LMPtypeId[AtCSum[mol]:AtCSum[mol + 1]] = _np.tile(sysTop.LMPtype[mol], sysTop.nMOL[mol])
        seq = _np.arange(m, m + sysTop.nMOL[mol])
        m = _np.max(seq)
        MOLid[AtCSum[mol]:AtCSum[mol + 1]] = _np.repeat(seq, sysTop.nBEAD[mol])
    sysName = '-'.join(sysTop.MOL)
    trajO = _cTRAJ(outTraj, 'w')
    trajI.read(nAtom=nAtom)
    print('\noutput trajectory: {}'.format(outTraj))
    n = 1
    m = 1
    if lastSnap:
        pass
    else:
        if end == -1:
            end = 1e10
        while True:
            if n >= begin and n <= end:
                if (n - begin) % stride == 0:
                    xs, ys, zs, x, y, z = trajI.rmPBC(MolId=sysTop.MOLnum, BondsMap=sysTop.BondsMap)
                    if trajI.time is not None:
                        trajO.time = trajI.time
                    else:
                        trajO.time = m - 1
                    if trajI.timeStep is not None:
                        trajO.timeStep = trajI.timeStep
                    else:
                        trajO.timeStep = m - 1
                    if m == 1:
                        if trajI.time is not None:
                            print('Time = {}'.format(trajO.time))
                        else:
                            print('TimeStep = {}'.format(trajO.timeStep))
                    elif m % 2 == 0:
                        if trajI.time is not None:
                            print('\rTime = {}'.format(trajO.time), end='')
                        else:
                            print('\rTimeStep = {}'.format(trajO.timeStep), end='')
                    if trajO.Type == 'LMP':
                        trajO.xs = xs
                        trajO.ys = ys
                        trajO.zs = zs
                    else:
                        trajO.x = x
                        trajO.y = y
                        trajO.z = z
                    trajO.nAtom = trajI.nAtom
                    trajO.atId = trajI.atId
                    if trajI.atType is None:
                        trajO.atType = LMPtypeId
                    else:
                        trajO.atType = trajI.atType
                    if trajI.resId is None:
                        trajO.resId = MOLid
                    else:
                        trajO.resId = trajI.resId
                    if trajI.resName is None:
                        trajO.resName = MOLname
                    else:
                        trajO.resName = trajI.resName
                    if trajI.atName is None:
                        trajO.atName = BEADname
                    else:
                        trajO.atName = trajI.atName
                    trajO.boxMat = trajI.boxMat
                    trajO.write(modelN=m, sysName=sysName)
                    m += 1
            elif n > end:
                trajI.close()
                trajO.close()
                break
            trajI.read(nAtom=nAtom)
            if trajI.eof:
                trajI.close()
                trajO.close()
                break
            n += 1
    if m > 1:
        print('\n\nnumber of snapshots : {}'.format(m - 1))
    Sec = round(_time.time() - start_time)
    Min, Sec = divmod(Sec, 60)
    Hour, Min = divmod(Min, 60)
    Hour, Min, Sec = int(Hour) , int(Min) , int(Sec)
    print('Processing Time: {0:d}:{1:02d}:{2:02d}'.format(Hour, Min, Sec))

#####################################################################
#####################################################################
#####################################################################

def RefinePot(setFile, LMP, MPI=None, nProc=None):
    '''This program reads an input options file, then refines the initial potential function through IBI or IMC methods.

Parameters:
• setFile: Input setting file.
• LMP: The path of the LAMMPS executable.
• MPI: The path of the mpirun executable. By default, the MPI executable is not used.
• nProc: The number of parallel processes. The default is None.

Outputs:
• The last configuration of iteration n (i-*.[output].data.out)
• The potential fuctions of iteration n (i-*.[output].pot)
• The potential updates of iteration n (i-*.[output].dpot)
• The distribution functions of iteration n (i-*.[output].hist or i-*.[output].dist)
• The simulation log file of iteration n (i-*.[output].log)
• The simulation trajectory file of iteration n (i-*.[output].lammpstrj or .xtc)
• The log file of this program ([output].SCG.log)'''

    ##########################################
    ##########################################

    def searchFile(sysName, ext):
        dirL = _glob.glob(sysName + '.*' + ext)
        suffix = []
        for line in dirL:
            temp = _Rstrip(line, '.' + ext).split('.')
            if len(temp) == 2:
                suffix.append(int(temp[1]))
        suffix.sort()
        nameList = [sysName + '.' + str(i) for i in suffix]
        if len(nameList) > 0:
            maxSuffix = max(suffix)
        else:
            maxSuffix = -1
        return nameList, maxSuffix

    def readOption(setFile):
        fid = open(setFile)
        optF = fid.readlines()
        fid.close()
        top = method = sysName = inPot = output = refHist = cutoff = iterate = None
        Temp = 300
        cutoffSkin = 2
        nEQsnaps = 0
        maxNB = 100
        maxB = 100
        maxA = 10
        binNB = 0.01
        binAD = 0.05
        corFactor = 0.1
        Lambda = 0.0
        dpotSmooth = 0.5
        potSmooth = 0.0
        pressTarget = 0
        pressFactor = -1
        corNB = corB = corA = corD = True
        lastConf = saveTraj = False
        for line in optF:
            lsp = line.split('#')[0]
            if 'TOP' in lsp.upper():
                top = lsp.split('=')[1].strip()
            elif 'METHOD' in lsp.upper():
                method = lsp.split('=')[1].strip()
            elif 'ITERATE' in lsp.upper():
                iterate = int(lsp.split('=')[1].strip())
            elif 'CORFACTOR' in lsp.upper():
                corFactor = float(lsp.split('=')[1].strip())
            elif 'LAMBDA' in lsp.upper():
                Lambda = float(lsp.split('=')[1].strip())
                if Lambda < 0:
                    mess = '"Lambda" must be >= 0'
                    raise _scg4pyError(mess)
            elif 'NEQSNAPS' in lsp.upper():
                nEQsnaps = int(lsp.split('=')[1].strip())
            elif 'DPOTSMOOTH' in lsp.upper():
                dpotSmooth = float(lsp.split('=')[1].strip())
            elif 'POTSMOOTH' in lsp.upper():
                potSmooth = float(lsp.split('=')[1].strip())
            elif 'PRESSTARGET' in lsp.upper():
                pressTarget = float(lsp.split('=')[1].strip())
            elif 'PRESSFACTOR' in lsp.upper():
                pressFactor = float(lsp.split('=')[1].strip())
            elif 'CORNB' in lsp.upper():
                if lsp.split('=')[1].strip().upper() == 'YES':
                    corNB = True
                elif lsp.split('=')[1].strip().upper() == 'NO':
                    corNB = False
            elif 'CORB' in lsp.upper():
                if lsp.split('=')[1].strip().upper() == 'YES':
                    corB = True
                elif lsp.split('=')[1].strip().upper() == 'NO':
                    corB = False
            elif 'CORA' in lsp.upper():
                if lsp.split('=')[1].strip().upper() == 'YES':
                    corA = True
                elif lsp.split('=')[1].strip().upper() == 'NO':
                    corA = False
            elif 'CORD' in lsp.upper():
                if lsp.split('=')[1].strip().upper() == 'YES':
                    corD = True
                elif lsp.split('=')[1].strip().upper() == 'NO':
                    corD = False
            elif 'CUTOFF' in lsp.upper() and not 'CUTOFFSKIN' in lsp.upper():
                cutoff = float(lsp.split('=')[1].strip())
            elif 'CUTOFFSKIN' in lsp.upper():
                cutoffSkin = float(lsp.split('=')[1].strip())
            elif 'MAXNB' in lsp.upper():
                maxNB = float(lsp.split('=')[1].strip())
            elif 'MAXB' in lsp.upper():
                maxB = float(lsp.split('=')[1].strip())
            elif 'MAXA' in lsp.upper():
                maxA = float(lsp.split('=')[1].strip())
            elif 'BINNB' in lsp.upper():
                binNB = float(lsp.split('=')[1].strip())
            elif 'BINAD' in lsp.upper():
                binAD = float(lsp.split('=')[1].strip())
            elif 'TEMP' in lsp.upper():
                Temp = float(lsp.split('=')[1].strip())
            elif 'INPOT' in lsp.upper():
                inPot = lsp.split('=')[1].strip()
            elif 'REFHIST' in lsp.upper():
                refHist = lsp.split('=')[1].strip()
            elif 'OUTPUT' in lsp.upper():
                output = lsp.split('=')[1].strip()
            elif 'LASTCONF' in lsp.upper():
                if lsp.split('=')[1].strip().upper() == 'YES':
                    lastConf = True
                elif lsp.split('=')[1].strip().upper() == 'NO':
                    lastConf = False
            elif 'SAVETRAJ' in lsp.upper():
                if lsp.split('=')[1].strip().upper() == 'YES':
                    saveTraj = True
                elif lsp.split('=')[1].strip().upper() == 'NO':
                    saveTraj = False
            elif 'SYSNAME' in lsp.upper():
                sysName = lsp.split('=')[1].strip()
        if method == 'IBI':
            Lambda = 0
        return [top, Temp, cutoff, cutoffSkin, maxNB, maxB, maxA, binNB, binAD, nEQsnaps, method,
                iterate, corFactor, Lambda, dpotSmooth, potSmooth, pressTarget, pressFactor,
                corNB, corB, corA, corD, inPot, refHist, output, lastConf, saveTraj, sysName]

    def checkSCGlog(SysName, trajExt, Output):
        SCGlogFile = Output + '.SCG.log'
        dataOut = SysName + '.data.out'
        restartFile = SysName + '.restart1'
        iteration = 1
        restart = False
        ended = False
        if _os.path.isfile(SCGlogFile):
            print('reading {} :'.format(SCGlogFile))
            fid = open(SCGlogFile)
            LogLines = fid.readlines()
            fid.close()
            for line in LogLines:
                lsp = line.split('#')[0]
                if 'Iteration' in lsp:
                    iteration = int(lsp.split('=')[1])
                print(line.strip('\n'))
                _time.sleep(0.02)
            prefix = 'i-' + str(iteration).zfill(3) + '.' + Output
            trajList, trajNum = searchFile(sysName, trajExt)
            if len(trajList) > 0 or _os.path.isfile(sysName + '.' + trajExt):
                checkTraj = True
            else:
                checkTraj = False
            if _os.path.isfile(restartFile) and checkTraj:
                restart = True
            elif _os.path.isfile(prefix + '.data') and checkTraj:
                ended = True
            if iteration == 1:
                LogLines = []
            elif iteration > 1:
                while 'Iteration' not in LogLines[-1]:
                    LogLines = LogLines[0:-1]
                while '###' not in LogLines[-1]:
                    LogLines = LogLines[0:-1]
            with open(SCGlogFile, 'w') as fid:
                for line in LogLines:
                    fid.write(line)
            print ('\n\n#####################\n#####################\n#####################\n\n')
            print('\nPotential refinement process is being continued from iteration {}\n'.format(iteration))
        else:
            _os.system('rm -f {}'.format(dataOut))
            _os.system('rm -f {0:s}*{1:s}'.format(sysName, trajExt))
            _os.system('rm -f {0:s}*{1:s}'.format(sysName, 'log'))
            _os.system('rm -f {}.restart*'.format(SysName))
            _os.system('rm -f i-*{}*'.format(Output))
        return iteration, restart, ended

    def devCalc(x, yTgt, y, normalize):
        dx = float(_np.mean(_np.diff(x)))
        if normalize == True:
            yTgt = yTgt / _np.trapz(yTgt, dx=dx)
            y = y / _np.trapz(y, dx=dx)
        dev = _np.trapz((yTgt - y) ** 2, x=x) / (_np.abs(x[-1] - x[0]))
        return dev

    def expSmoothing(x, y, sigma):
        if sigma > 0:
            SIG = sigma * _np.mean(_np.diff(x))
            yy = _np.zeros(_np.size(x), dtype=float)
            for i in _np.arange(_np.size(yy)):
                expX = _np.exp((-1 * (x[i] - x) ** 2) / (2 * SIG ** 2))
                Z = _np.sum(expX)
                yy[i] = _np.sum(y * expX) / Z
            y_out = yy
        else:
            y_out = y
        return y_out

    def IBIupdating(x, df, dfRef, pot, kB, Temp, corFactor, dpotSmooth, potSmooth):
        dpot = kB * Temp * (_np.log(df) - _np.log(dfRef)) * corFactor
        dpot = expSmoothing(x, dpot, dpotSmooth)
        pot = pot + dpot
        pot = expSmoothing(x, pot, potSmooth)
        return dpot, pot

    def IMCupdating(x, dPotVec, j, l, pot, corFactor, dpotSmooth, potSmooth):
        dpot = dPotVec[j: j + l] * corFactor
        dpot = expSmoothing(x, dpot, dpotSmooth)
        pot = pot + dpot
        pot = expSmoothing(x, pot, potSmooth)
        j = j + l
        return j, dpot, pot

    def WriteLog(FileName, String):
        fid = open(FileName, 'a')
        fid.write(String)
        fid.close()

    ############################################################
    kB = 1.987204e-03
    top, Temp, cutoff, cutoffSkin, maxNB, maxB, maxA, binNB, binAD, nEQsnaps, method,\
    iterate, corFactor, Lambda, dpotSmooth, potSmooth, pressTarget, pressFactor,\
    corNB, corB, corA, corD, inPot, refHist, output, lastConf, saveTraj, sysName = readOption(setFile)
    if cutoff <= cutoffSkin:
        mess = '\ncutoff <= cutoffSkin'
        raise _scg4pyError(mess)
    HIST = DIST = False
    if refHist.endswith('.hist'):
        HIST = True
    elif refHist.endswith('.dist'):
        DIST = True
    else:
        mess = 'Refernce file "{}" must be one of ".hist" or ".dist" types.'.format(refHist)
        raise _scg4pyError(mess)
    script = sysName + '.lmp.in'
    dataIn = sysName + '.data.in'
    dataOut = sysName + '.data.out'
    SCGlogFile = output + '.SCG.log'
    fid = open(script)
    scList = fid.readlines()
    fid.close()
    dumpLine = ''
    for line in scList:
        lsp = line.split('#')[0]
        if 'dump' in lsp and 'dump_' not in lsp:
            dumpLine = lsp
            break
    if '.mpiio.lammpstrj' in dumpLine:
        trajExt = 'mpiio.lammpstrj'
    elif '.lammpstrj' in dumpLine:
        trajExt = 'lammpstrj'
    elif '.xtc' in dumpLine:
        trajExt = 'xtc'
    else:
        mess = 'supported trajectories: ".mpiio.lammpstrj", ".lammpstrj", or ".xtc".'
        raise _scg4pyError(mess)
    iteration, restart, ended = checkSCGlog(sysName, trajExt, output)
    refHistTab = _cTAB(refHist, 'r')
    itnList = range(iteration, iterate + 1)
    if iteration == 1:
        inPotTab = _cTAB(inPot, 'r')
    else:
        prefix = 'i-' + str(iteration - 1).zfill(3) + '.' + output
        inPotTab = _cTAB(prefix + '.pot', 'r')
        _os.system('cp -f {0:s} {1:s}'.format(prefix + '.data', dataIn))
    if not (_np.all(refHistTab.BondName == inPotTab.BondName) and
            _np.all(refHistTab.BondType == inPotTab.BondType)):
        mess = 'Bond terms of the "{0:s}" file are not complible with "{1:s}"'.format(refHist, inPot)
        raise _scg4pyError(mess)
    if not (_np.all(refHistTab.AngleName == inPotTab.AngleName) and
            _np.all(refHistTab.AngleType == inPotTab.AngleType)):
        mess = 'Angle terms of the "{0:s}" file are not complible with "{1:s}"'.format(refHist, inPot)
        raise _scg4pyError(mess)
    if not (_np.all(refHistTab.DihedralName == inPotTab.DihedralName) and
            _np.all(refHistTab.DihedralType == inPotTab.DihedralType)):
        mess = 'Dihedral terms of the "{0:s}" file are not complible with "{1:s}"'.format(refHist, inPot)
        raise _scg4pyError(mess)
    if not _np.all(refHistTab.NonBondType == inPotTab.NonBondType):
        mess = 'Non-Bonded terms of the "{0:s}" file are not complible with "{1:s}"'.format(refHist, inPot)
        raise _scg4pyError(mess)
    fid = open(dataIn)
    inData = fid.readlines()
    fid.close()
    lx = ly = lz = nAtoms =  0
    nBonds = nAngles = nDihedrals = 0
    for line in inData:
        lsp = line.split('#')[0]
        if 'atoms' in lsp:
            nAtoms = int(lsp.split()[0])
        elif 'bonds' in lsp:
            nBonds = int(lsp.split()[0])
        elif 'angles' in lsp:
            nAngles = int(lsp.split()[0])
        elif 'dihedrals' in lsp:
            nDihedrals = int(lsp.split()[0])
        elif 'xlo' in lsp:
            s = lsp.split()
            lx = abs(float(s[1]) - float(s[0]))
        elif 'ylo' in lsp:
            s = lsp.split()
            ly = abs(float(s[1]) - float(s[0]))
        elif 'zlo' in lsp:
            s = lsp.split()
            lz = abs(float(s[1]) - float(s[0]))
        elif 'Atoms' in lsp:
            break
    if lx > 0 and ly > 0 and lz > 0 and nAtoms > 0:
        Volume = lx * ly * lz
    else:
        mess = 'error in parsing data file "{}".'.format(dataIn)
        raise _scg4pyError(mess)
    sysTop = _cTOP(top)
    sysTop.SetBondedIndex()
    nB = 0
    for i in range(len(sysTop.BONDtypeSet)):
        nB += len(sysTop.BONDtypeSetIdx[i])
    if nB != nBonds:
        mess = 'number of bonds in the "{0:s}" file is different from the "{1:s}" file.'.format(top, dataIn)
        raise _scg4pyError(mess)
    nA = 0
    for i in range(len(sysTop.ANGLEtypeSet)):
        nA += len(sysTop.ANGLEtypeSetIdx[i])
    if nA != nAngles:
        mess = 'number of angles in the "{0:s}" file is different from the "{1:s}" file.'.format(top, dataIn)
        raise _scg4pyError(mess)
    nD = 0
    for i in range(len(sysTop.DIHEDRALtypeSet)):
        nD += len(sysTop.DIHEDRALtypeSetIdx[i])
    if nD != nDihedrals:
        mess = 'number of dihedrals in the "{0:s}" file is different from the "{1:s}" file.'.format(top, dataIn)
        raise _scg4pyError(mess)
    sysTop.calcBondsMap()
    sysTop.SetNonBondedIndex()
    nBeads = _np.sum(sysTop.nMOL * sysTop.nBEAD)
    del nB, nA, nD, nBonds, nAngles, nDihedrals
    if nBeads != nAtoms:
        mess = 'number of atoms in data file "{0:s}" is not equal to top file "{1:s}"'.format(dataIn, top)
        raise _scg4pyError(mess)
    if not (_np.all(refHistTab.BondName == sysTop.BONDtypeSetName) and
            _np.all(refHistTab.BondType == sysTop.BONDtypeSet)):
        mess = 'Bond terms of the "{0:s}" file are not complible with tabulated potentials.'.format(top)
        raise _scg4pyError(mess)
    if not (_np.all(refHistTab.AngleName == sysTop.ANGLEtypeSetName) and
            _np.all(refHistTab.AngleType == sysTop.ANGLEtypeSet)):
        mess = 'Angle terms of the "{0:s}" file are not complible with tabulated potentials.'.format(top)
        raise _scg4pyError(mess)
    if not (_np.all(refHistTab.DihedralName == sysTop.DIHEDRALtypeSetName) and
            _np.all(refHistTab.DihedralType == sysTop.DIHEDRALtypeSet)):
        mess = 'Dihedral terms of the "{0:s}" file are not complible with tabulated potentials.'.format(top)
        raise _scg4pyError(mess)
    if not _np.all(refHistTab.NonBondType == sysTop.NonBONDED_Set):
        mess = 'Non-Bonded terms of the "{0:s}" file are not complible with tabulated potentials.'.format(top)
        raise _scg4pyError(mess)
    nPairType = len(sysTop.NonBONDED_Set)
    nBondType = len(sysTop.BONDtypeSet)
    nAngleType = len(sysTop.ANGLEtypeSet)
    nDihedralType = len(sysTop.DIHEDRALtypeSet)
    NBref = _np.zeros([nPairType, 3], dtype=object) # r, rdf, hist
    NBpot = _np.zeros(nPairType, dtype=object)
    for i in range(nPairType):
        r = refHistTab.NonBondX[i]
        hbin = 0.5 * _np.mean(_np.diff(r))
        rHist = _np.concatenate([[r[0] - hbin], r[0:-1] + hbin, [r[-1] + hbin]])
        jacob = (4. / 3) * _np.pi * (rHist[1:] ** 3 - rHist[0:-1] ** 3)
        rdf = refHistTab.NonBondY[i]
        nIntract = len(sysTop.NonBONDED_SetIdx[i])
        Hist = (jacob * rdf * nIntract) / Volume
        rdf[rdf <= 1e-10] = 1e-10
        Hist[Hist <= 1e-10] = 1e-10
        maxInd = len(r[r <= (cutoff - cutoffSkin)])
        NBref[i, 0] = r[0: maxInd]
        NBref[i, 1] = rdf[0: maxInd]
        NBref[i, 2] = Hist[0: maxInd]
        rPot = inPotTab.NonBondX[i]
        try:
            rPot = rPot[0: maxInd]
            if not _np.all(rPot == r[0: maxInd]):
                name = 'Non-Bonded_{0:s}-{1:s}'.format(refHistTab.NonBondType[i, 0], refHistTab.NonBondType[i, 1])
                mess = '{0:s}: {1:s} and {2:s} are incompatible'.format(name, inPot, refHist)
                raise _scg4pyError(mess)
            NBpot[i] = inPotTab.NonBondY[i][0: maxInd]
        except:
            name = 'Non-Bonded_{0:s}-{1:s}'.format(refHistTab.NonBondType[i, 0], refHistTab.NonBondType[i, 1])
            mess = '{0:s}: {1:s} and {2:s} are incompatible'.format(name, inPot, refHist)
            raise _scg4pyError(mess)

    Bref = _np.zeros([nBondType, 2], dtype=object)  # r, hist
    for i in range(nBondType):
        Bref[i, 0] = refHistTab.BondX[i]
        bdf = refHistTab.BondY[i]
        bdf[bdf <= 1e-10] = 1e-10
        Bref[i, 1] = bdf
        if not _np.all(refHistTab.BondX[i] == inPotTab.BondX[i]):
            name = 'Bond_{0:s}-{1:s}'.format(inPotTab.BondName[i, 0], inPotTab.BondName[i, 1])
            mess = '{0:s}: {1:s} and {2:s} are incompatible'.format(name, inPot, refHist)
            raise _scg4pyError(mess)

    Aref = _np.zeros([nAngleType, 2], dtype=object)  # r, hist
    for i in range(nAngleType):
        Aref[i, 0] = refHistTab.AngleX[i]
        adf = refHistTab.AngleY[i]
        adf[adf <= 1e-10] = 1e-10
        Aref[i, 1] = adf
        if not _np.all(refHistTab.AngleX[i] == inPotTab.AngleX[i]):
            name = 'Angle_{0:s}-{1:s}-{2:s}'.format(inPotTab.AngleName[i, 0], inPotTab.AngleName[i, 1],
                                                    inPotTab.AngleName[i, 2])
            mess = '{0:s}: {1:s} and {2:s} are incompatible'.format(name, inPot, refHist)
            raise _scg4pyError(mess)

    Dref = _np.zeros([nDihedralType, 2], dtype=object)  # r, hist
    for i in range(nDihedralType):
        Dref[i, 0] = refHistTab.DihedralX[i]
        ddf = refHistTab.DihedralY[i]
        ddf[ddf <= 1e-10] = 1e-10
        Dref[i, 1] = ddf
        if not _np.all(refHistTab.DihedralX[i] == inPotTab.DihedralX[i]):
            name = 'Dihedral_{0:s}-{1:s}-{2:s}-{3:s}'.format(inPotTab.DihedralName[i, 0], inPotTab.DihedralName[i, 1],
                                                             inPotTab.DihedralName[i, 2], inPotTab.DihedralName[i, 3])
            mess = '{0:s}: {1:s} and {2:s} are incompatible'.format(name, inPot, refHist)
            raise _scg4pyError(mess)
    Bpot = inPotTab.BondY
    Apot = inPotTab.AngleY
    Dpot = inPotTab.DihedralY
    NBtab = _np.zeros([nPairType, 3], dtype=object)  # X, POT, FORCE (extrapolated)
    Btab = _np.zeros([nBondType, 3], dtype=object)  # X, POT, FORCE (extrapolated)
    Atab = _np.zeros([nAngleType, 3], dtype=object)  # X, POT, FORCE (extrapolated)
    Dtab = _np.zeros([nDihedralType, 3], dtype=object)  # X, POT, FORCE (extrapolated)

    print ('\n#####################\n')
    for itn in itnList:
        if not (corB or corA or corD or corNB):
            mess = 'Error: none of iteraction selected for potential refinement.'
            raise _scg4pyError(mess)
        print ('Iteration = {}'.format(str(itn)))
        WriteLog(SCGlogFile, '\nIteration = {}\n'.format(str(itn)))
        prefix = 'i-' + str(itn).zfill(3) + '.' + output
        if itn == 1:
            p0 = 'i-' + str(0).zfill(3) + '.' + output
            _os.system('cp -f ' + dataIn + ' ' + p0 + '.data')
            _os.system('cp -f ' + inPot + ' ' + p0 + '.pot')

        outPotTab = _cTAB(prefix + '.pot', 'w')
        outPotTab.BondType = refHistTab.BondType
        outPotTab.BondName = refHistTab.BondName
        outPotTab.AngleType = refHistTab.AngleType
        outPotTab.AngleName = refHistTab.AngleName
        outPotTab.DihedralType = refHistTab.DihedralType
        outPotTab.DihedralName = refHistTab.DihedralName
        outPotTab.NonBondType = refHistTab.NonBondType

        outdPotTab = _cTAB(prefix + '.dpot', 'w')
        outdPotTab.BondType = refHistTab.BondType
        outdPotTab.BondName = refHistTab.BondName
        outdPotTab.AngleType = refHistTab.AngleType
        outdPotTab.AngleName = refHistTab.AngleName
        outdPotTab.DihedralType = refHistTab.DihedralType
        outdPotTab.DihedralName = refHistTab.DihedralName
        outdPotTab.NonBondType = refHistTab.NonBondType

        if HIST:
            outHistTab = _cTAB(prefix + '.hist', 'w')
        else:
            outHistTab = _cTAB(prefix + '.dist', 'w')
        outHistTab.BondType = refHistTab.BondType
        outHistTab.BondName = refHistTab.BondName
        outHistTab.AngleType = refHistTab.AngleType
        outHistTab.AngleName = refHistTab.AngleName
        outHistTab.DihedralType = refHistTab.DihedralType
        outHistTab.DihedralName = refHistTab.DihedralName
        outHistTab.NonBondType = refHistTab.NonBondType

        outPotTab.BondX = Bref[:, 0]
        outPotTab.BondY = Bpot
        for i in range(nBondType):
            X, Pot, Force = outPotTab.extrpPot_Bond(i, binNB, maxB, cutoff)
            Btab[i, 0] = X
            Btab[i, 1] = Pot
            Btab[i, 2] = Force
        outPotTab.AngleX = Aref[:, 0]
        outPotTab.AngleY = Apot
        for i in range(nAngleType):
            X, Pot, Force = outPotTab.extrpPot_Angle(i, binAD, maxA)
            Atab[i, 0] = X
            Atab[i, 1] = Pot
            Atab[i, 2] = Force
        outPotTab.DihedralX = Dref[:, 0]
        outPotTab.DihedralY = Dpot
        for i in range(nDihedralType):
            X, Pot, Force = outPotTab.extrpPot_Dihedral(i, binAD)
            Dtab[i, 0] = X
            Dtab[i, 1] = Pot
            Dtab[i, 2] = Force
        outPotTab.NonBondX = NBref[:, 0]
        outPotTab.NonBondY = NBpot
        for i in range(nPairType):
            X, Pot, Force = outPotTab.extrpPot_NonBonded(i, binNB, maxNB, cutoff, cutoffSkin)
            NBtab[i, 0] = X
            NBtab[i, 1] = Pot
            NBtab[i, 2] = Force
        _writeLMPtab(sysName, Btab, outPotTab.BondType, Atab, outPotTab.AngleType,
                     Dtab, outPotTab.DihedralType, NBtab, outPotTab.NonBondType, refinePot=True)
        ### LAMMPS running
        if not ended:
            runLMP(sysName, LMP, MPI, nProc, restart, 10)
            if _os.path.isfile(dataOut):
                _os.system('cp -f {0:s} {1:s}.data'.format(dataOut, prefix))
                _os.system('rm -f {}.restart*'.format(sysName))
                if lastConf:
                    _os.system('mv -f {0:s} {1:s}'.format(dataOut, dataIn))
                else:
                    _os.system('rm -f {}'.format(dataOut))
            else:
                mess = '"{}" was not generated'.format(dataOut)
                raise _scg4pyError(mess)
        ended = False
        restart = False
        trajList, trajNum = searchFile(sysName, trajExt)
        trajNum += 1
        if _os.path.isfile(sysName + '.' + trajExt):
            _os.rename(sysName + '.' + trajExt, sysName + '.' + str(trajNum) + '.' + trajExt)
            trajList.append(sysName + '.' + str(trajNum))
        logList, logNum = searchFile(sysName, 'log')
        logNum += 1
        if _os.path.isfile(sysName + '.log'):
            _os.rename(sysName + '.' + 'log', sysName + '.' + str(logNum) + '.log')
            logList.append(sysName + '.' + str(logNum))
        if len(logList) > 1:
            fidO = open(prefix + '.log', 'w')
            for i in range(len(logList)):
                fidI = open(logList[i] + '.log')
                lines = fidI.read()
                fidI.close()
                fidO.write(lines + '\n')
            fidO.close()
        else:
            _os.system('cp -f {0:s}.log {1:s}.log'.format(logList[0], prefix))
        logIt = parseLMPLog(prefix + '.log', timeStep=1, plot=False)
        pressIt = logIt[-2][nEQsnaps:]
        if len(pressIt) == 0:
            mess = '"nEQsnaps" > number of total snapshots'
            raise _scg4pyError(mess)
        pressIt = _np.mean(pressIt)
        print('   Average of the pressure = {0:.4f} [Atmosphere]'.format(pressIt))
        WriteLog(SCGlogFile, '   Average of the pressure = {0:.4} [Atmosphere]\n'.format(pressIt))
        if pressFactor >= 0:
            print('   Pressure correction factor = {}'.format(pressFactor))
            WriteLog(SCGlogFile, '   Pressure correction factor = {}\n'.format(pressFactor))
        sMat = 0
        if corNB:
            for i in range(nPairType):
                sMat += len(NBref[i, 0])
        if corB:
            for i in range(nBondType):
                sMat += len(Bref[i, 0])
        if corA:
            for i in range(nAngleType):
                sMat += len(Aref[i, 0])
        if corD:
            for i in range(nDihedralType):
                sMat += len(Dref[i, 0])
        SaSb = _np.zeros([sMat, sMat], dtype=float)
        SaSbVec = _np.zeros([1, sMat], dtype=float)
        Sa_x_SbVec = _np.zeros([1, sMat], dtype=float)
        dSVec = _np.zeros(sMat, dtype=float)
        NBrdf = _np.zeros(nPairType, dtype=object)
        NBhist = _np.zeros(nPairType, dtype=object)
        NBdpot = _np.zeros(nPairType, dtype=object)
        NBdev = _np.zeros([nPairType, 2], dtype=float) # dist and hist decviation
        Bhist = _np.zeros(nBondType, dtype=object)
        Bdpot = _np.zeros(nBondType, dtype=object)
        Bdev = _np.zeros([nBondType, 2], dtype=float) # dist and hist decviation
        Ahist = _np.zeros(nAngleType, dtype=object)
        Adpot = _np.zeros(nAngleType, dtype=object)
        Adev = _np.zeros([nAngleType, 2], dtype=float) # dist and hist decviation
        Dhist = _np.zeros(nDihedralType, dtype=object)
        Ddpot = _np.zeros(nDihedralType, dtype=object)
        Ddev = _np.zeros([nDihedralType, 2], dtype=float) # dist and hist decviation
        Volume = 0
        T1 = _time.time()
        readedT = -1
        nSnap = 0
        nSnapTot = 0
        print ('   Parsing the trajectory to calculate potential updates')
        print ('      Method = {}'.format(method))
        WriteLog(SCGlogFile, '   Method = {}\n'.format(method))
        sysDFs = _cDFs()
        for t in range(len(trajList)):
            trajName = trajList[t] + '.' + trajExt
            Traj = _cTRAJ(trajName, 'r')
            while True:
                Traj.read(nAtoms)
                if Traj.eof:
                    Traj.close()
                    break
                if Traj.time is not None:
                    currentT = Traj.time
                else:
                    currentT = Traj.timeStep
                if currentT > readedT:
                    readedT = currentT
                    nSnapTot += 1
                    if nSnapTot > nEQsnaps:
                        Volume += Traj.boxMat[0] * Traj.boxMat[1] * Traj.boxMat[2]
                        if Traj.xu is not None:
                            x, y, z = Traj.xu, Traj.yu, Traj.zu
                            PairData = sysDFs.calcNB(x, y, z, sysTop.NonBONDED_SetIdx, Traj.lmpBox, unwrapped=True)
                            BondData = sysDFs.calcB(x, y, z, sysTop.BONDtypeSetIdx)
                            AngleData = sysDFs.calcA(x, y, z, sysTop.ANGLEtypeSetIdx)
                            DihedralData = sysDFs.calcD(x, y, z, sysTop.DIHEDRALtypeSetIdx)
                        elif Traj.x is not None or Traj.xs is not None:
                            xs, ys, zs, x, y, z = Traj.rmPBC(sysTop.MOLnum, sysTop.BondsMap)
                            PairData = sysDFs.calcNB(xs, ys, zs, sysTop.NonBONDED_SetIdx, Traj.boxMat)
                            BondData = sysDFs.calcB(x, y, z, sysTop.BONDtypeSetIdx)
                            AngleData = sysDFs.calcA(x, y, z, sysTop.ANGLEtypeSetIdx)
                            DihedralData = sysDFs.calcD(x, y, z, sysTop.DIHEDRALtypeSetIdx)
                        else:
                            mess = 'Error in parsing the trajectory'
                            raise _scg4pyError(mess)
                        j = 0
                        for i in range(nPairType):
                            r = refHistTab.NonBondX[i]
                            hbin = 0.5 * _np.mean(_np.diff(r))
                            rHist = _np.concatenate([[r[0] - hbin], r[0:-1] + hbin, [r[-1] + hbin]])
                            Hist = _np.histogram(PairData[i], bins=rHist)[0]
                            NBhist[i] = NBhist[i] + Hist
                            if corNB and method == 'IMC':
                                l = len(NBref[i, 0])
                                SaSbVec[0, j: j + l] = Hist[0: l]
                                j = j + l
                        for i in range(nBondType):
                            r = Bref[i, 0]
                            hbin = 0.5 * _np.mean(_np.diff(r))
                            rHist = _np.concatenate([[r[0] - hbin], r[0:-1] + hbin, [r[-1] + hbin]])
                            Hist = _np.histogram(BondData[i], bins=rHist)[0]
                            Bhist[i] = Bhist[i] + Hist
                            if corB and method == 'IMC':
                                l = len(Bref[i, 0])
                                SaSbVec[0, j: j + l] = Hist[0: l]
                                j = j + l
                        for i in range((nAngleType)):
                            theta = Aref[i, 0]
                            hbin = 0.5 * _np.mean(_np.diff(theta))
                            tHist = _np.concatenate([[theta[0] - hbin], theta[0:-1] + hbin, [theta[-1] + hbin]])
                            Hist = _np.histogram(AngleData[i], bins=tHist)[0]
                            Ahist[i] = Ahist[i] + Hist
                            if corA and method == 'IMC':
                                l = len(Aref[i, 0])
                                SaSbVec[0, j: j + l] = Hist[0: l]
                                j = j + l
                        for i in range((nDihedralType)):
                            phi = Dref[i, 0]
                            hbin = 0.5 * _np.mean(_np.diff(phi))
                            pHist = _np.concatenate([[phi[0] - hbin], phi[0:-1] + hbin, [phi[-1] + hbin]])
                            Hist = _np.histogram(DihedralData[i], bins=pHist)[0]
                            Dhist[i] = Dhist[i] + Hist
                            if corD and method == 'IMC':
                                l = len(Dref[i, 0])
                                SaSbVec[0, j: j + l] = Hist[0: l]
                                j = j + l
                        if method == 'IMC':
                            SaSb = SaSb + _np.dot(SaSbVec.T, SaSbVec)
                        nSnap += 1
        NBhist /= nSnap
        Bhist /= nSnap
        Ahist /= nSnap
        Dhist /= nSnap
        Volume /= nSnap
        j = 0
        for i in range(nPairType):
            NBhist[i][NBhist[i] <= 1e-10] = 1e-10
            l = len(NBref[i, 0])
            r = NBref[i, 0]
            if corNB and method == 'IBI':
                Hist = NBhist[i][0: l]
                HistRef = NBref[i, 2]
                pot = NBpot[i]
                NBdpot[i], NBpot[i] = IBIupdating(r, Hist, HistRef, pot, kB, Temp, corFactor, dpotSmooth, potSmooth)
                if pressFactor >= 0:
                    deltaP = pressIt - pressTarget
                    A = -0.1 * kB * Temp * _np.sign(deltaP) * min(1, abs(pressFactor * deltaP))
                    deltaU = A * (1 - (r / r[-1]))
                    NBpot[i] += deltaU
            elif corNB and method == 'IMC':
                Sa_x_SbVec[0, j: j + l] = NBhist[i][0: l]
                dSVec[j: j + l] = NBref[i, 2] - NBhist[i][0: l]
                j = j + l
            elif corNB == False:
                NBdpot[i] = _np.zeros(l, dtype=float)
            r = refHistTab.NonBondX[i]
            hbin = 0.5 * _np.mean(_np.diff(r))
            rHist = _np.concatenate([[r[0] - hbin], r[0:-1] + hbin, [r[-1] + hbin]])
            nIntract = len(sysTop.NonBONDED_SetIdx[i])
            jacob = (4. / 3) * _np.pi * (rHist[1:] ** 3 - rHist[0:-1] ** 3)
            NBrdf[i] = (Volume / nIntract) * (NBhist[i] / jacob)
            NBdev[i, 0] = devCalc(r[0: l], NBref[i, 1], NBrdf[i][0: l], normalize=False)
            NBdev[i, 1] = devCalc(r[0: l], NBref[i, 2], NBhist[i][0: l], normalize=False)

        for i in range(nBondType):
            Bhist[i][Bhist[i] <= 1e-10] = 1e-10
            l = len(Bref[i, 0])
            r = Bref[i, 0]
            Bdev[i, 0] = devCalc(r, Bref[i, 1], Bhist[i], normalize=True)
            if DIST:
                Bdev[i, 1] = Bdev[i, 0]
            else:
                Bdev[i, 1] = devCalc(r, Bref[i, 1], Bhist[i], normalize=False)
            if corB and method == 'IBI':
                if HIST:
                    df = Bhist[i]
                else:
                    df = Bhist[i] / _np.trapz(y=Bhist[i], x=r)
                dfRef = Bref[i, 1]
                pot = Bpot[i]
                Bdpot[i], Bpot[i] = IBIupdating(r, df, dfRef, pot, kB, Temp, corFactor, dpotSmooth, potSmooth)
            elif corB and method == 'IMC':
                Sa_x_SbVec[0, j: j + l] = Bhist[i]
                if HIST:
                    dSVec[j: j + l] = Bref[i, 1] - Bhist[i]
                else:
                    dSVec[j: j + l] = Bref[i, 1] - (Bhist[i] / _np.trapz(y=Bhist[i], x=r))
                j = j + l
            elif corB == False:
                Bdpot[i] = _np.zeros(l, dtype=float)

        for i in range(nAngleType):
            Ahist[i][Ahist[i] <= 1e-10] = 1e-10
            l = len(Aref[i, 0])
            theta = Aref[i, 0]
            Adev[i, 0] = devCalc(theta, Aref[i, 1], Ahist[i], normalize=True)
            if DIST:
                Adev[i, 1] = Adev[i, 0]
            else:
                Adev[i, 1] = devCalc(theta, Aref[i, 1], Ahist[i], normalize=False)
            if corA and method == 'IBI':
                if HIST:
                    df = Ahist[i]
                else:
                    df = Ahist[i] / _np.trapz(y=Ahist[i], x=theta)
                dfRef = Aref[i, 1]
                pot = Apot[i]
                Adpot[i], Apot[i] = IBIupdating(theta, df, dfRef, pot, kB, Temp, corFactor, dpotSmooth, potSmooth)
            elif corA and method == 'IMC':
                Sa_x_SbVec[0, j: j + l] = Ahist[i]
                if HIST:
                    dSVec[j: j + l] = Aref[i, 1] - Ahist[i]
                else:
                    dSVec[j: j + l] = Aref[i, 1] - (Ahist[i] / _np.trapz(y=Ahist[i], x=theta))
                j = j + l
            elif corA == False:
                Adpot[i] = _np.zeros(l, dtype=float)

        for i in range(nDihedralType):
            Dhist[i][Dhist[i] <= 1e-10] = 1e-10
            l = len(Dref[i, 0])
            phi = Dref[i, 0]
            Ddev[i, 0] = devCalc(phi, Dref[i, 1], Dhist[i], normalize=True)
            if DIST:
                Ddev[i, 1] = Ddev[i, 0]
            else:
                Ddev[i, 1] = devCalc(phi, Dref[i, 1], Dhist[i], normalize=False)
            if corD and method == 'IBI':
                if HIST:
                    df = Dhist[i]
                else:
                    df = Dhist[i] / _np.trapz(y=Dhist[i], x=phi)
                dfRef = Dref[i, 1]
                pot = Dpot[i]
                Ddpot[i], Dpot[i] = IBIupdating(phi, df, dfRef, pot, kB, Temp, corFactor, dpotSmooth, potSmooth)
            elif corD and method == 'IMC':
                Sa_x_SbVec[0, j: j + l] = Dhist[i]
                if HIST:
                    dSVec[j: j + l] = Dref[i, 1] - Dhist[i]
                else:
                    dSVec[j: j + l] = Dref[i, 1] - (Dhist[i] / _np.trapz(y=Dhist[i], x=phi))
                j = j + l
            elif corD == False:
                Ddpot[i] = _np.zeros(l, dtype=float)

        if method == 'IMC':
            SaSb = SaSb / nSnap
            Sa_x_Sb = _np.dot(Sa_x_SbVec.T, Sa_x_SbVec)
            beta = 1 / (kB * Temp)
            CovMat = -1 * beta * (SaSb - Sa_x_Sb)
            svdvals = _linalg.svdvals(CovMat)
            if 0 < Lambda <= 1:
                LAMBDA = Lambda * _np.max(svdvals)
            elif Lambda > 1:
                LAMBDA = Lambda
            else:
                LAMBDA = 0
            noiseMat = LAMBDA * _np.eye(_np.size(CovMat, axis=0))
            TikhonovReg = _np.dot(_np.linalg.pinv(_np.dot(CovMat.T, CovMat) + noiseMat), CovMat.T)
            dPotVec = _np.dot(TikhonovReg, dSVec)
            j = 0
            if corNB:
                for i in range(nPairType):
                    l = len(NBref[i, 0])
                    r = NBref[i, 0]
                    pot = NBpot[i]
                    j, NBdpot[i], NBpot[i] = IMCupdating(r, dPotVec, j, l, pot, corFactor, dpotSmooth, potSmooth)
                    if pressFactor >= 0:
                        deltaP = pressIt - pressTarget
                        A = -0.1 * kB * Temp * _np.sign(deltaP) * min(1, abs(pressFactor * deltaP))
                        deltaU = A * (1 - (r / r[-1]))
                        NBpot[i] += deltaU
            if corB:
                for i in range(nBondType):
                    l = len(Bref[i, 0])
                    r = Bref[i, 0]
                    pot = Bpot[i]
                    j, Bdpot[i], Bpot[i] = IMCupdating(r, dPotVec, j, l, pot, corFactor, dpotSmooth, potSmooth)
            if corA:
                for i in range(nAngleType):
                    l = len(Aref[i, 0])
                    theta = Aref[i, 0]
                    pot = Apot[i]
                    j, Adpot[i], Apot[i] = IMCupdating(theta, dPotVec, j, l, pot, corFactor, dpotSmooth, potSmooth)
            if corD:
                for i in range(nDihedralType):
                    l = len(Dref[i, 0])
                    phi = Dref[i, 0]
                    pot = Dpot[i]
                    j, Ddpot[i], Dpot[i] = IMCupdating(phi, dPotVec, j, l, pot, corFactor, dpotSmooth, potSmooth)
        T2 = _time.time()
        Sec = round((T2 - T1))
        Min, Sec = divmod(Sec, 60)
        Hour, Min = divmod(Min, 60)
        Hour, Min, Sec = int(Hour), int(Min), int(Sec)
        print ('      Processing Time = {0:d}:{1:02d}:{2:02d}'.format(Hour, Min, Sec))
        print ('   Total number of snapshots = {}'.format(nSnapTot))
        print ('   Number of snapshots used in the potential refinement = {}'.format(nSnap))
        WriteLog(SCGlogFile, '   Processing Time = {0:d}:{1:02d}:{2:02d}\n'.format(Hour, Min, Sec))
        WriteLog(SCGlogFile, '   Total number of the snapshots = {}\n'.format(nSnapTot))
        WriteLog(SCGlogFile, '   Number of snapshots used in the potential refinement = {}\n'.format(nSnap))

        if saveTraj:
            print ('   Output trajectory : {0:s}.{1:s}'.format( prefix, trajExt))
            if len(trajList) == 1:
                _os.rename(trajList[0] + '.' + trajExt, prefix + '.' + trajExt)
            else:
                trajList = [(i + '.' + trajExt) for i in trajList]
                trajCat(trajList, prefix + '.' + trajExt, logging=False)
        _os.system('rm -f {0:s}*{1:s}'.format(sysName, trajExt))

        outPotTab.NonBondX = NBref[:, 0]
        outPotTab.NonBondY = NBpot
        outPotTab.BondX = Bref[:, 0]
        outPotTab.BondY = Bpot
        outPotTab.AngleX = Aref[:, 0]
        outPotTab.AngleY = Apot
        outPotTab.DihedralX = Dref[:, 0]
        outPotTab.DihedralY = Dpot
        outPotTab.write()

        outdPotTab.NonBondX = NBref[:, 0]
        outdPotTab.NonBondY = NBdpot
        outdPotTab.BondX = Bref[:, 0]
        outdPotTab.BondY = Bdpot
        outdPotTab.AngleX = Aref[:, 0]
        outdPotTab.AngleY = Adpot
        outdPotTab.DihedralX = Dref[:, 0]
        outdPotTab.DihedralY = Ddpot
        outdPotTab.write()

        outHistTab.NonBondX = refHistTab.NonBondX
        outHistTab.NonBondY = NBrdf
        outHistTab.BondX = Bref[:, 0]
        outHistTab.BondY = Bhist
        outHistTab.AngleX = Aref[:, 0]
        outHistTab.AngleY = Ahist
        outHistTab.DihedralX = Dref[:, 0]
        outHistTab.DihedralY = Dhist
        outHistTab.write()

        if method == 'IMC':
            with open(prefix + '.svd', 'w') as svdFile:
                print ('   Singular values of the covariance matrix : {}.svd'.format(prefix))
                for i in range(len(svdvals)):
                    svdFile.write('%f\n' % svdvals[i])

        if method == 'IMC':
            print ('   Covariance Matrix Statistics:')
            print ('      Minimum = {:.4f}'.format(_np.min(CovMat)))
            print ('      Maximum = {:.4f}'.format(_np.max(CovMat)))
            print ('      Mean , stdv = {:.4f} , {:.4f}'.format(_np.mean(CovMat), _np.std(CovMat)))
            print ('      Number of Singular Values = {:d}'.format(len(svdvals)))
            print ('      Minimum of Singular Values = {:.4f}'.format(_np.min(svdvals)))
            print ('      Maximum of Singular Values = {:.4f}'.format(_np.max(svdvals)))
            print ('      Tikhonov Regularization Lambda = {:.4f}'.format(LAMBDA))
            WriteLog(SCGlogFile, '   Covariance Matrix Statistics:\n')
            WriteLog(SCGlogFile, '      Minimum = {:.4f}\n'.format(_np.min(CovMat)))
            WriteLog(SCGlogFile, '      Maximum = {:.4f}\n'.format(_np.max(CovMat)))
            WriteLog(SCGlogFile, '      Mean , stdv = {:.4f} , {:.4f}\n'.format(_np.mean(CovMat), _np.std(CovMat)))
            WriteLog(SCGlogFile, '      Number of Singular Values = {:d}\n'.format(len(svdvals)))
            WriteLog(SCGlogFile, '      Minimum of Singular Values = {:.4f}\n'.format(_np.min(svdvals)))
            WriteLog(SCGlogFile, '      Maximum of Singular Values = {:.4f}\n'.format(_np.max(svdvals)))
            WriteLog(SCGlogFile, '      Tikhonov Regularization Lambda = {:.4f}\n'.format(LAMBDA))

        B = []
        C = []
        WriteLog(SCGlogFile, '   Deviation between current and target distributions :\n')
        if nPairType != 0:
            a = 'Non-Bondeds'
            b = _np.sum(NBdev[:, 0])
            c = _np.sum(NBdev[:, 1])
            B = _np.append(B, b)
            C = _np.append(C, c)
            print ('      {0:13} : ( dist = {1:.5e} , hist = {2:.5e} )'.format(a, b, c))
            WriteLog(SCGlogFile,'      {0:13} : ( dist = {1:.5e} , hist = {2:.5e} )\n'.format(a, b, c))
        if nBondType != 0:
            a = 'Bonds'
            b = _np.sum(Bdev[:, 0])
            c = _np.sum(Bdev[:, 1])
            B = _np.append(B, Bdev)
            C = _np.append(C, Bdev)
            print ('      {0:13} : ( dist = {1:.5e} , hist = {2:.5e} )'.format(a, b, c))
            WriteLog(SCGlogFile,'      {0:13} : ( dist = {1:.5e} , hist = {2:.5e} )\n'.format(a, b, c))
        if nAngleType != 0:
            a = 'Angles'
            b = _np.sum(Adev[:, 0])
            c = _np.sum(Adev[:, 1])
            B = _np.append(B, Adev)
            C = _np.append(C, Adev)
            print ('      {0:13} : ( dist = {1:.5e} , hist = {2:.5e} )'.format(a, b, c))
            WriteLog(SCGlogFile,'      {0:13} : ( dist = {1:.5e} , hist = {2:.5e} )\n'.format(a, b, c))
        if nDihedralType != 0:
            a = 'Dihedrals'
            b = _np.sum(Ddev[:, 0])
            c = _np.sum(Ddev[:, 1])
            B = _np.append(B, Ddev)
            C = _np.append(C, Ddev)
            print ('      {0:13} : ( dist = {1:.5e} , hist = {2:.5e} )'.format(a, b, c))
            WriteLog(SCGlogFile,'      {0:13} : ( dist = {1:.5e} , hist = {2:.5e} )\n'.format(a, b, c))
        a = 'Total'
        print ('      {0:13} : ( dist = {1:.5e} , hist = {2:.5e} )'.format(a, _np.sum(B), _np.sum(C)))
        WriteLog(SCGlogFile, '      {0:13} : ( dist = {1:.5e} , hist = {2:.5e} )\n'.format(a, _np.sum(B), _np.sum(C)))
        print ('#####################\n')
        WriteLog(SCGlogFile, '#####################\n\n')

#####################################################################
#####################################################################
#####################################################################
