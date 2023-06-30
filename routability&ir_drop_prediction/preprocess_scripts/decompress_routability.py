import os
import time

t = time.time()
decompress_path = '../routability_features_decompressed'
print('Create decompress dir')
os.system('mkdir -p %s && rm -rf %s*' % (decompress_path, decompress_path))

filelist = os.walk('../routability_features')
for parent,dirnames,filenames in filelist:
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.gz':
            filepath = os.path.join(parent, filename)
            print('Process %s' %(filename))
            os.system('gzip -fdk %s' % filepath)
            os.system('mkdir -p %s ' % (parent.replace('routability_features','routability_features_decompressed')))
            os.system('tar -xf %s -C %s && rm -f %s' % (filepath.replace('.gz',''), \
            parent.replace('routability_features','routability_features_decompressed'), filepath.replace('.gz','')))

print('Decompress finished in %ss'%(time.time() - t))
