import os
import time

t = time.time()
decompress_path = '../IR_drop_features_decompressed'
print('Create decompress dir.')
os.system('mkdir -p %s && rm -rf %s/*' % (decompress_path, decompress_path))
os.system('cat ../IR_drop_features/power_t*.tar.gz*.* > ../IR_drop_features/power_t.tar.gz')

filelist = os.walk('../IR_drop_features')
for parent,dirnames,filenames in filelist:
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.gz':
            filepath = os.path.join(parent, filename)
            print('Process %s.' %(filename))
            os.system('gzip -dk %s' % filepath)
            os.system('tar -xf %s -C %s && rm -f %s' % (filepath.replace('.gz',''), \
            parent.replace('IR_drop_features','IR_drop_features_decompressed'), filepath.replace('.gz','')))

print('Decompress finished in %ss'%(time.time() - t))
