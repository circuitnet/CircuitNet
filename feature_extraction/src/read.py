import os, re, bisect, gzip
import numpy as np
import pandas as pd
from src.util import save
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class Paraser:
    def __init__(self, root_dir, arg, save_name):
        self.root_dir = root_dir
        self.save_name = save_name
        self.save_path = arg.save_path
        self.plot = arg.plot
        
        self.coordinate_x = []
        self.coordinate_y = []
        self.IR_drop_map = None
        self.total_power_map = None
        self.eff_res_VDD_map = None
        self.instance_count = None
        self.instance_IR_drop = None

    def get_IR_drop_features(self):
        die_size = (1000, 1000)
        self.IR_drop_map = np.zeros(die_size)
        self.instance_count = np.zeros(die_size)
        self.instance_IR_drop = np.empty(die_size, dtype=object)
        for i in np.ndindex(die_size):
            self.instance_IR_drop[i] = []
        self.instance_name = np.empty(die_size, dtype=object)
        for i in np.ndindex(die_size):
            self.instance_name[i] = []
        self.total_power_map = np.zeros(die_size)
        self.eff_res_VDD_map = np.zeros(die_size)

        self.coordinate_x = np.arange(0,die_size[0],1.44)
        self.coordinate_y = np.arange(0,die_size[1],1.152)      # based on the row height from LEF

        try:
            if 'nvdla' in self.root_dir:
                data_power = pd.read_csv(os.path.join(self.root_dir, 'NV_nvdla.inst.power.rpt'),sep='\s+',header=1)
            else:
                data_power = pd.read_csv(os.path.join(self.root_dir, 'pulpino_top.inst.power.rpt'),sep='\s+',header=1)
            data_r = pd.read_csv(os.path.join(self.root_dir, 'eff_res.rpt'),sep='\s+', low_memory=False)
            data_ir = pd.read_csv(os.path.join(self.root_dir, 'static_ir'),sep='\s+')
        except Exception as e:
            print('one of the report not exists')
            return 0       
        
        # parse static_ir 因为report开头的井号被当作单独一列，这里的key也往前移了一位。
        # 即data_ir['inst_vdd']对应的数据是vdd_drop，而不是inst_vdd。vdd_drop被作为预测的target。
        # 类似地，data_ir['pwr_net']对应的数据是location，而不是pwr_net，data_ir['location']对应的数据是instance name，而不是location。
        ir = data_ir['inst_vdd']     
        location = data_ir['pwr_net'] 
        name = data_ir['location']

        max_x = 0
        max_y = 0
        for i,j,k in zip(location, ir, name):
            x, y = i.split(',')
            gcell_x = bisect.bisect_left(self.coordinate_x, float(x)-10)
            gcell_y = bisect.bisect_left(self.coordinate_y, float(y)-10)
            if gcell_x > max_x:
                max_x = gcell_x
            if gcell_y > max_y: 
                max_y = gcell_y
            if j > self.IR_drop_map[gcell_x, gcell_y]:
                self.IR_drop_map[gcell_x, gcell_y] = j
            self.instance_count[gcell_x, gcell_y] += 1

            self.instance_IR_drop[gcell_x, gcell_y].append(j)
            self.instance_name[gcell_x, gcell_y].append(k)

        self.IR_drop_map = self.IR_drop_map[0:max_x+1,0:max_y+1]
        self.instance_count = self.instance_count[0:max_x+1,0:max_y+1]
        self.instance_IR_drop = np.concatenate(self.instance_IR_drop.ravel())
        self.instance_name = np.concatenate(self.instance_name.ravel())
        save(self.save_path, 'features/IR_drop', self.save_name, self.IR_drop_map)
        save(self.save_path, 'features/instance_count', self.save_name, self.instance_count)
        save(self.save_path, 'features/instance_IR_drop', self.save_name, self.instance_IR_drop)
        save(self.save_path, 'features/instance_name', self.save_name, self.instance_name)

        # parse inst.power.rpt
        power = data_power['total_power']
        bbox = data_power['bbox']

        for i,j in zip(bbox, power):
            x1, y1, x2, y2 = i[1:-1].split(',')
            x = (float(x1)+float(x2))/2
            y = (float(y1)+float(y2))/2
            gcell_x = bisect.bisect_left(self.coordinate_x, float(x)-10)
            gcell_y = bisect.bisect_left(self.coordinate_y, float(y)-10)
            self.total_power_map[gcell_x, gcell_y] += j
        self.total_power_map = self.total_power_map[0:max_x+1,0:max_y+1]

        save(self.save_path, 'features/total_power', self.save_name, self.total_power_map)

        # parse eff_res.rpt
        vdd_r = data_r['loop_r']
        location_x = data_r['gnd_r']
        location_y = data_r['vdd(x']

        for i,j,k in zip(location_x, location_y, vdd_r):
            x = i[1:]
            y = j
            if i == '-' or j == '-' or k == '-':
                continue
            gcell_x = bisect.bisect_left(self.coordinate_x, float(x)-10)
            gcell_y = bisect.bisect_left(self.coordinate_y, float(y)-10)
            self.eff_res_VDD_map[gcell_x, gcell_y] += float(k)

        self.eff_res_VDD_map = self.eff_res_VDD_map[0:max_x+1,0:max_y+1]

        save(self.save_path, 'features/eff_res_VDD', self.save_name, self.eff_res_VDD_map)
        # for visualization
        if self.plot:
            ir_max = self.IR_drop_map.max()
            ir_mean = self.IR_drop_map.mean()
            
            r_max = self.eff_res_VDD_map.max()
            r_mean = self.eff_res_VDD_map.mean()

            self.save_name = '{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(self.save_name, ir_max, ir_mean, r_max, r_mean)


            print(self.save_name)
            if ir_max > 5*ir_mean:
                print(os.path.join(self.save_path, 'visual', self.save_name ))
                if not os.path.exists(os.path.join(self.save_path, 'visual2')):
                    os.makedirs(os.path.join(self.save_path, 'visual2'))
                fig = sns.heatmap(data=self.IR_drop_map, cmap="rainbow").get_figure()
                fig.savefig(os.path.join(self.save_path,'visual2', '{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.png'.format(self.save_name, ir_max, ir_mean, r_max, r_mean)), dpi=100)
                plt.close()
                self.save_name = 'check/' + self.save_name


            if not os.path.exists(os.path.join(self.save_path, 'visual', self.save_name)):
                os.makedirs(os.path.join(self.save_path, 'visual', self.save_name))
            fig = sns.heatmap(data=self.eff_res_VDD_map, cmap="rainbow").get_figure()
            fig.savefig(os.path.join(self.save_path,'visual', self.save_name,'eff_res_VDD.png'), dpi=100)
            plt.close()

            fig = sns.heatmap(data=self.total_power_map, cmap="rainbow").get_figure()
            fig.savefig(os.path.join(self.save_path,'visual', self.save_name,'total_power.png'), dpi=100)
            plt.close()

            fig = sns.heatmap(data=self.IR_drop_map, cmap="rainbow").get_figure()
            fig.savefig(os.path.join(self.save_path,'visual', self.save_name,'IR_drop.png'), dpi=100)
            plt.close()
