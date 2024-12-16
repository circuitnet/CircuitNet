# coding=utf-8
import os, gzip, re
import multiprocessing
from multiprocessing import Process

def divide_n(list_in, n):
    list_out = [ [] for i in range(n)]
    for i,e in enumerate(list_in):
        list_out[i%n].append(e)
    return list_out

def fix_module_name(read_list, save_dir):
    module_to_fix = ('adv_dbg_if', 'axi_spi_slave_wrap', 'periph_bus_wrap', 'apb_uart', 
    'apb_spi_master', 'apb_timer', 'apb_event_unit', 'apb_i2c', 'apb_fll_if', 'apb_pulpino')
    for read_path in read_list:
        READ_COMPONENTS = False
        save_name = os.path.basename(read_path)
        save_path = os.path.join(save_dir, save_name)
        clock = os.path.basename(read_path).split('-')[-5].split('c')[-1]
        with gzip.open(read_path, 'rt') as read_file:
            with gzip.open(save_path, 'wt') as write_file:
                for line in read_file:
                    if line.startswith("COMPONENTS"):
                        READ_COMPONENTS = True
                        continue
                    elif "END COMPONENTS" in line:
                        READ_COMPONENTS = False
                        continue
                    if READ_COMPONENTS:
                        ret = re.match(r'^\-\s.*?\s(.*?)\s\+.*$', line)
                        if ret:
                            stdcell_name = ret.group(1)
                            prefix = stdcell_name.rsplit('_',1)[0]
                            if stdcell_name.startswith(module_to_fix):
                                line = re.sub(stdcell_name, f'{prefix}_{clock}', line)
                    write_file.writelines(line)

                    
if __name__ == '__main__':
    def_dir = './DEF'
    save_dir = './DEF-fix'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    read_list = [os.path.join(def_dir, i) for i in os.listdir(def_dir)]

    nlist = divide_n(read_list, 10)
    process = []
    multiprocessing.set_start_method("spawn", force=True)
    for l in nlist:
        p = Process(target=fix_module_name, args=(l, save_dir))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()
