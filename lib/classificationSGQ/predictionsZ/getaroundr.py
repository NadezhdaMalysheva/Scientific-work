#!/home/horungev/opt/anaconda3/bin/python3

'''
python3 getaroundr.py -r 30.0 -i /home/horungev/NWAY/test/COSMOS_XMM.fits -index ID -iRA RA -iDEC DEC -cat gaia sdss galex -merge -o Re_getaroundr.fits 
'''
#remove it after 
import time
import sys
#if '/home/horungev/opt/anaconda3/bin' not in sys.path:
#	sys.path.append('/home/horungev/opt/anaconda3/bin')
#
if '/home/horungev/.local/lib/python3.6/site-packages/' not in sys.path:
	sys.path.append('/home/horungev/.local/lib/python3.6/site-packages/')

if True: #manual setting path
	import astropy
	import healpy as hp
	from astropy.table import Table
	import fitsio
	
else:
	'/home/horungev/.local/lib/python3.6/site-packages/healpy'
	'/home/horungev/.local/lib/python3.6/site-packages/astropy'

import argparse
import numpy as np
from numpy.lib.recfunctions import append_fields,stack_arrays,merge_arrays,rename_fields
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
import shutil

import multiprocessing

Vsyspath = '/home/horungev/Catalogs/SRG/crossmatch'
sys.path.append(Vsyspath)
from libgetinradec import Dcat,rfieldslist,rindexfields,query_polygon,readjobcatfile,user_obj_list,cropinRADECbox,freadfitstable,sdss_readindex,calcRADECboxarea,appendcolumnsnway,getinradecBOX


Lcat = list(Dcat.keys())
lcat = '\n'.join(Lcat)

Ltimeit=[time.time()]

def listfiles(heal_indexes,Aindex_fields,fields_list_df,a_str,Vcat):
	if Aindex_fields.shape[0]==1: #this done for healpix array
        	Aindex_fields = Aindex_fields.T
	healpix_fields_numbers = Aindex_fields[heal_indexes]
	HFN_df_unique = np.unique(healpix_fields_numbers.flatten())
	if HFN_df_unique[0] == 0: # Избавляюсь от нуля, получаю серийные номера нужных фитсов
    		HFN_df_unique = HFN_df_unique[1:]
	
	fields_list_filtered = fields_list_df.loc[fields_list_df['SERIAL_NUMBER'].isin(HFN_df_unique)]
	#fields_path_array = a_str + '/' + fields_list_filtered.loc[ :, 'FILENAME']
	#fields_list_filtered['SERIAL_NUMBER']=fields_list_filtered['SERIAL_NUMBER'].map(tuple)
	return fields_list_filtered
	

def correlatecatinrad(ra1,dec1,ra2,dec2,radius):
	cc = SkyCoord(ra=ra1*u.degree,dec=dec1*u.degree)
	ccmatch = SkyCoord(ra=ra2*u.degree, dec = dec2*u.degree)
	idcc, idmatch, d2d, d3d = ccmatch.search_around_sky(cc, radius*u.deg/3600.0)
	return idcc, idmatch, d2d

def appendcolumnnamesuffix(A,suffx):
	Lnames = A.dtype.names
	for i in range(len(Lnames)):
		A=rename_fields(A,{Lnames[i]:Lnames[i]+suffx})
	return A

def renamecolumnnamesuffix(A,Lnames,Lrenames):
	Drename={}
	for i in range(len(Lnames)):
		if Lrenames[i]!=None:
			Drename[Lnames[i]]=Lrenames[i]
	rename_fields(A,Drename)
	return A
	

def matchaftercorrelation(cat1,cat2):
	Lnames1 = cat1.dtype.names
	Lnames2 = cat2.dtype.names
	Lnames0 = Lnames1

	'''
	cnt=-1
	while any(np.isin(Lnames0,Lnames2)):
		cnt = cnt + 1
		Lnames0 = [x+str(cnt) for x in Lnames1]
	if cnt>-1:
		for i in range(len(Lnames0)):
			rename_fields(cat1,{Lnames1[i]:Lnames0[i]})
	'''

	ncol1 = len(Lnames1)
	ncol2 = len(Lnames2)
	
	ncols = ncol1 + ncol2
	nrows = len(cat1)
	Ltype = np.dtype(cat1.dtype.descr+cat2.dtype.descr)
	print (nrows,len(cat1),len(cat2))
	b = np.zeros(nrows, dtype = Ltype)
	for s in cat1.dtype.names:
		b[s] = cat1[s]
	for s in cat2.dtype.names:
		b[s] = cat2[s]
	return b
	

	
	
		
	

def getaroundr(L):
	Ltimeit=[time.time()]
	Vjobfile,nside,icat,Vcat,hpicat,radius,Vsuffix,Vsepname=L
	Djob = readjobcatfile(Vjobfile)
	flist = rfieldslist(Djob['fieldslist']) #read fields list
	# read index_fields 
	if type(Djob['indexfieldsfile']) == str: #single files
		index_fields = rindexfields(Djob['indexfieldsfile'])
	elif type(Djob['indexfieldsfile']) == list: #splitted file in complicated case sdss
		index_fields = sdss_readindex(Djob['indexfieldsfile'])

	if 'Lshort' in Djob.keys():
		Lshort = Djob['Lshort']
	else:
		Lshort = None

	glfiles = lambda x: listfiles(x,index_fields,flist,Djob['datadir'],Vcat)['SERIAL_NUMBER']
	
	hpicat['pathind'] = hpicat['hpicat'].map(glfiles)
	hpicat['pathind'] = hpicat['pathind'].map(tuple)
	hpicat['npath'] = hpicat['pathind'].map(len)

	dnpath = pd.DataFrame()
	Lind=[]
	for i in range(hpicat['npath'].max()):
		dnpath[str(i)] = np.full(len(hpicat),-999)
		dnpath.loc[hpicat['npath']>i,str(i)]=hpicat.loc[hpicat['npath']>i,'pathind'].map(lambda x:x[i])
		Lind.append(np.unique(dnpath.loc[:,str(i)].values))

	Lindsum=[]
	Lstack = []
	for i in range(len(Lind)-1,-1,-1):
		#print (Lind[i])
		Lindsum.append(Lind[i])
		for j in range(Lind[i].size):
			if Lind[i][j]==-999: #exclude empty index
				continue
			ifield = Lind[i][j]-1 # index of catalog file 
			ifile = os.path.join(Djob['datadir'], flist.iloc[ifield]['FILENAME']) #path to catalog file
			print (flist.iloc[ifield]['FILENAME'], flist.iloc[ifield]['SERIAL_NUMBER'],ifield)
			mask = np.any(dnpath==Lind[i][j],axis=1).values #mask of path array
			
			if not os.path.exists(ifile):
				print (ifile,'catalog file not found')
				raise ValueError
			
			ind1 = hpicat.iloc[:,0].values[mask]
			ra1 = hpicat.iloc[:,1].values[mask]
			dec1 = hpicat.iloc[:,2].values[mask]
			ix1 = np.array(hpicat.index.to_numpy())[mask]

			tcat = freadfitstable(ifile, Lfields = Lshort)
			#mask bad ra dec #change here
			maskradec = np.logical_and(tcat[Djob['nRA']]!=-999,tcat[Djob['nDEC']]!=-999)
			tcat=tcat[maskradec]
			tcat= tcat.byteswap().newbyteorder()
			if Vcat=='sdss':
				mask = tcat['MODE']==1
				if not any(mask):
					continue
				else:
					tcat=tcat[mask]

			ra2 = tcat[Djob['nRA']]
			dec2 = tcat[Djob['nDEC']]
			idcc, idmatch, d2d = correlatecatinrad(ra1,dec1,ra2,dec2,radius)

			'''
			if isinstance(Vsuffix,str): # rename column names in correlated catalog
				if len(Vsuffix)>0:
					tcat=appendcolumnnamesuffix(tcat,Vsuffix) 
				else:
					tcat=appendcolumnnamesuffix(tcat,'_'+Vcat) # rename column names 
			'''

			cat1=icat[ix1[idcc]]
			cat1 = append_fields(cat1,names=Vsepname,data=d2d.to_value(u.arcsec))
			if isinstance(Vsuffix,str): # rename column names for input cat
				if len(Vsuffix)>0:
					cat1=appendcolumnnamesuffix(cat1,Vsuffix) 

			cat2=tcat[idmatch]
	
			b = matchaftercorrelation(cat1,cat2)
			Lstack.append(b)

		for j in range(i):
			mask = np.isin(Lind[j],Lind[i])
			Lind[j]=Lind[j][~mask]
		dnpath.drop(str(i),axis=1,inplace=True)
		print (dnpath.shape)
	
	if len(Lstack) == 0: #append at 2020/11/16
		print ('W A R N I N G ')
		print ('There is no object in the area of the requested catalog')
		print ('You could increase radius or try another object list')
		# create empty file
		ifile = os.path.join(Djob['datadir'], flist.iloc[0]['FILENAME'])
		tcatmock = freadfitstable(ifile, Lfields = Lshort)
		Aout = np.array([],dtype=tcatmock.dtype)	
	else:
		Aout = stack_arrays(Lstack,autoconvert=True)
	
	return (Aout)
	'''
	#print (dnpath,dnpath.shape)
	
	#hpicat['pathind'] = hpicat['pathind'].map(tuple)
	
	#print (hpicat['pathind'])
	#grouped = hpicat.groupby('pathind')
	#print (grouped.groups.keys())
	'''


	

def readinputcat(Vfile,nindex,nRA,nDEC,Dparam):
	hdu = fitsio.FITS(Vfile,case_sensitive = False)
	Lcol = hdu[1].get_colnames()
	
	Linputnames = []
	if Dparam['storefull']: #select strings to read full catalog
		for s in Lcol:
			if s in [nindex,nRA,nDEC]:
				continue
			Linputnames.append(s)
	
	if nindex in Lcol:
		data = hdu[1].read(columns=[nindex,nRA,nDEC]+Linputnames)
	else:
		data = hdu[1].read(columns=[nRA,nDEC]+Linputnames)
		data = np.lib.recfunctions.rec_append_fields(data,nindex,np.arange(len(data))+1)
		Lfirstcol = [nindex,nRA,nDEC]
		'''
		if nindex in Linputnames:
			Lfirstcol = [nindex,nRA,nDEC]
		else:
			Lfirstcol = [nRA,nDEC]
		'''
		data = data[Lfirstcol+Linputnames]

	hdu.close()
	data = data.byteswap().newbyteorder()
	#and rename dublicate column names
	'''
	Dnames={}
	Anames = data.dtype.names
	Dnamesrenames={}
	for keyn in Anames:
		if keyn.lower() in Dnames:
			ncnt = 0
			keynn = keyn.lower()+'_'+str(ncnt)
			while keynn in Dnames:
				ncnt = ncnt + 1
				keynn = keyn.lower()+'_'+str(ncnt)
			Dnamesrenames[keyn] = keynn
		else:
			Dnames[keyn.lower()]=keyn
	if len(Dnamesrenames)>0:
		print ('Rename following columns')
		print (Dnamesrenames)
		data = rename_fields(data,Dnamesrenames)
	'''

	return data

def readradecfromstr(ix,vra,vdec,nindex,nRA,nDEC):
	#data = np.array([(ix,), (vra,), (vdec,)], dtype=[(nindex, np.int64), (nRA,'<f8'), (nDEC,'<f8')])
	data = np.recarray((1,), dtype = [(nindex, np.int64), (nRA,'<f8'), (nDEC,'<f8')])
	data[nindex] = ix
	data[nRA] = vra
	data[nDEC] = vdec
	return data

def qd(A,L):
	return hp.query_disc(L[0], vec=A, radius=L[1], nest=True, inclusive=True)

def hpindisk(Nside,ra,dec,rad):
	cc = SkyCoord(ra*u.deg,dec*u.deg)
	g = lambda j: np.sort(hp.query_disc(Nside, vec=(j.cartesian.x,j.cartesian.y,j.cartesian.z), radius=np.deg2rad(rad/3600.0), nest=True, inclusive=True))
	heal_indexes=list(map(g,cc[:]))
	return heal_indexes

def checkrepeatingcolumns(A,Vsuffix):
	Lnames = A.dtype.names #check repeating columns names
	Ishift=2
	for i in range(len(Lnames)):
		if i>Ishift:
			break
		if Lnames[i].upper() in (x.upper() for x in Lnames[Ishift:]) :
			for j in range(Ishift):
				A=rename_fields(A,{Lnames[j]:Lnames[j]+Vsuffix})
	return A
			

def worker(procnum, func, vargs, return_dict): #for multiprocessing
	return_dict[procnum] = func(vargs)	

def savefile(Vout,args,Vcat,Aout):
	if '.fits' in Vout:
		if len(args.cat)>1:
			Vout1 = Vout.replace('.fits', '_'+Vcat+'.fits')
		else:
			Vout1 = Vout
			ft = fitsio.FITS(Vout1,'rw',case_sensitive=True,clobber=True)
			ft.case_sensitive=True
			ft.write(Aout,case_sensitive=True)
			ft.close()
			print (Vout1,'saved')
	else:
			print ('the output file name must have .fits extenstion')		

def main():
	Vstr='\n to rename column in the output table you can write eg -index ID:IDx; -RA ra:RA0 -DEC dec:DEC0'
	Vsepnamedef='sep'
	Ltimeit.append(time.time())
	Vtime0 = time.time()
	#argument parsing
	parser = argparse.ArgumentParser(description="select objects in certain radius around catalog")
	parser.add_argument('-i',nargs='?',help='<Required> input catalog')
	parser.add_argument('-index',nargs='?',type=str,default='ix',help='<Required> column name for input cat index'+Vstr)
	parser.add_argument('-iRA',nargs='?',type=str,default='RA',help='column name for input cat right ascension'+Vstr)
	parser.add_argument('-iDEC',nargs='?',type=str,default='DEC',help='column name for input cat declination'+Vstr)
	parser.add_argument('-iSEPNAME',nargs='?',type=str,default=Vsepnamedef,help='column name for separation -iSEPNAME :sep')

	parser.add_argument('-r',nargs='?',help='<Required> search radius in arcsec')
	parser.add_argument('-c',nargs='+',help='Coordinates of circle 216.00 +35.94',default=[])
	parser.add_argument('-cat',nargs='+',choices=Lcat,help=f'<Required> select catalog from following list:\n {lcat}')
	parser.add_argument('-NSIDE',nargs='*',type=int,default=2048,help='NSIDE healpix parameter')
	parser.add_argument('-o',nargs='?',type=str,default='output.fits',help='output file of catalog')
	parser.add_argument('-merge',action='store_true',default=False,help='merge all correlations into single file')
	parser.add_argument('-full',action='store_true',default=False,help='store all columns from input catalog to output')
	parser.add_argument('-asfx',nargs='?',action='append',type=str,help='add suffix to input catalog')
	parser.add_argument('-mpi',action='store_true',default=False,help='switch on an mpi mode')
	args = parser.parse_args()
	print (args) #prints arguments

	Nside = int(args.NSIDE)
	Vinput = args.i #check existing input file
	Lcc = args.c
	Vrad = float(args.r)
	Vout = args.o
	mpimode=args.mpi
	Dparam = {'storefull':args.full}


	if args.asfx == None: #set suffix to append catalog
		Vsuffix=None
	else:
		if isinstance(args.asfx,list):
			if args.asfx[0]==None:
				Vsuffix=''
			else:
				Vsuffix=args.asfx[0]
	

	Lbasecol = [args.index,args.iRA,args.iDEC,args.iSEPNAME]
	#configure column if need to rename
	gparse0 = lambda x:x.split(':')[0] if ':' in x else x
	Lbasecolparse = [gparse0(x) for x in Lbasecol]
	Vindex,ViRA,ViDE,Vsepname = Lbasecolparse
	if len(Vsepname)==0:
		Vsepname=Vsepnamedef
		Lbasecolparse[3]=Vsepname
	gparse1 = lambda x:x.split(':')[1] if ':' in x else None
	Lbasecolrenamed=[gparse1(x) for x in Lbasecol]
	
	if len(Lcc)==0:
		if not os.path.exists(Vinput):
			print ('input file not found -i',Vinput)
			raise ValueError
			
		icat = readinputcat(Vinput,Vindex,ViRA,ViDE,Dparam)
		if icat.size==0: #write empty file
			print ('W A R N I N G')
			print ('input file is empty, has zero rows')
			shutil.copy(Vinput,Vout)
			print (Vout,'saved')
			return None

			

	elif len(Lcc)==2 or len(Lcc)==3:
		vra,vdec=Lcc[-2:]
		if len(Lcc)==3:
			vix = int(Lcc[0])
		else:
			vix=1
			
		icat = readradecfromstr(vix,float(vra),float(vdec),Vindex,ViRA,ViDE)
	else:
		print ('SET RA and DEC by -c flag, e.g.:')
		print ('-c 180.0 +63')
		print ('or with index')
		print ('-c 1 180.0 +63')
		raise ValueError

	hpicat = hpindisk(Nside,icat[ViRA],icat[ViDE],Vrad)

	Lkeyscat = [Vindex,ViRA,ViDE]
	df = pd.DataFrame(data=icat[Lkeyscat])
	df['hpicat']=hpicat
	if mpimode:
		hpicatmpi = hp.ang2pix(64,icat[ViRA],icat[ViDE],lonlat=True)
		Lsetmpi = list(set(hpicatmpi))
		Lworkmpi = []
		for i in range(len(Lsetmpi)):
			mask = hpicatmpi == Lsetmpi[i]
			Lworkmpi.append(mask)
	
	if any(Lbasecolrenamed):
		icat=renamecolumnnamesuffix(icat,Lbasecolparse,Lbasecolrenamed)
	
	
	#groups = pd.groupby('hpicat')
	for Vcat in args.cat:
		if not Vcat in list(Dcat.keys()):
			print (Vcat,'not found in list catalogs', Dcat.keys())
		Vjobfile=Dcat[Vcat]
		
		
		if mpimode: #correlation in mpi mode
			manager = multiprocessing.Manager()
			return_dict = manager.dict()
			jobs = []
			if True: #uncomment for testing
				for i in range(len(Lworkmpi)):
					msk = Lworkmpi[i]
					#getaroundr((Vjobfile,Nside,icat,Vcat,df,Vrad,Vsuffix,Vsepname))
					print (type(df),type(df[msk]),len(df[msk]),len(df))
					Ltmp = (Vjobfile,Nside,icat[msk],Vcat,df[msk],Vrad,Vsuffix,Vsepname)
					worker(i,getaroundr,Ltmp,return_dict)
					print (return_dict)

			raise
			for i in range(len(Lworkmpi)): #run mpi
				msk = Lworkmpi[i]
				Ltmp = (Vjobfile,Nside,icat[msk],Vcat,df[msk],Vrad,Vsuffix,Vsepname) #(Vjobfile,Nside,icat[msk],Vcat,df[msk],Vrad,Vsuffix,Vsepname)
				p = multiprocessing.Process(target=worker,args=(i,getaroundr,Ltmp,return_dict))
				jobs.append(p)
				p.start()
			for proc in jobs:
				proc.join()
			print (return_dict)
			raise
			
		else:
			Lgetinr=(Vjobfile,Nside,icat,Vcat,df,Vrad,Vsuffix,Vsepname)
			Aout = getaroundr(Lgetinr) #make a correlation 
		#Aout = checkrepeatingcolumns(Aout,'ix')
		if False: #set True for test 
			cc = SkyCoord(ra=Aout['RA_0']*u.degree,dec=Aout['DEC_0']*u.degree)
			ccmatch = SkyCoord(ra=Aout['RA']*u.degree, dec = Aout['DEC']*u.degree)
			sep = cc.separation(ccmatch).arcsecond
			for i in range(sep.size):
				print (Aout['sep_0'][i],sep[i])
			print (np.all(Aout['sep_0'] == sep))

			
	
		# write output:
		print (Aout.dtype.names)
		savefile(Vout,args,Vcat,Aout)
		'''
		if '.fits' in Vout:
			if len(args.cat)>1:
				Vout1 = Vout.replace('.fits', '_'+Vcat+'.fits')
			else:
				Vout1 = Vout
			ft = fitsio.FITS(Vout1,'rw',case_sensitive=True,clobber=True)
			ft.case_sensitive=True
			ft.write(Aout,case_sensitive=True)
			ft.close()
			print (Vout1,'saved')
		else:
			print ('the output file name mast have .fits extenstion')
		'''
	Ltimeit.append(time.time())
	print (np.diff(Ltimeit),np.sum(np.diff(Ltimeit)),time.time()-Vtime0,'Execution time')
		
		

if __name__ == "__main__":
	main()
