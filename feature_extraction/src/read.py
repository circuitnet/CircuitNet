import os, re, bisect, gzip
import numpy as np
from src.util import instance_direction_rect, instance_direction_bottom_left, save, my_range, is_gzip_file

""" 
read LEF file, extract geometric information of cells and pins and save them in lef_dict. 

:param path: path to LEF file.
:param lef_dict: empty dict or dict from the other LEF.
:param unit: distance convert factors, e.g., unit = 1000, then 1000 in DEF equals 1 micron in LEF.
:return: lef_dict {cell_name:{pin:{pin_name:pin_rect}, 'size':[unit*w,unit*h]}}
"""
def read_lef(path, lef_dict, unit):
    with open(path, 'r') as read_file:
        cell_name = ''
        pin_name = ''
        rect_list_left = []
        rect_list_lower = []
        rect_list_right = []
        rect_list_upper = []
        READ_MACRO = False
        for line in read_file:
            if line.lstrip().startswith('MACRO'):
                READ_MACRO = True
                cell_name = line.split()[1]
                lef_dict[cell_name] = {}
                lef_dict[cell_name]['pin'] = {}


            if READ_MACRO:
                if line.lstrip().startswith('SIZE'):
                    l = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
                    lef_dict[cell_name]['size'] = [unit * float(l[0]), unit * float(l[1])] # size [unit*w,unit*h]

                elif line.lstrip().startswith('PIN'):
                    pin_name = line.split()[1]


                elif line.lstrip().startswith('RECT'):
                    l = line.split()
                    rect_list_left.append(float(l[1]))
                    rect_list_lower.append(float(l[2]))
                    rect_list_right.append(float(l[3]))
                    rect_list_upper.append(float(l[4]))
                
                elif line.lstrip().startswith('END %s\n' % pin_name):
                    rect_left = min(rect_list_left) * unit
                    rect_lower = min(rect_list_lower) * unit
                    rect_right = max(rect_list_right) * unit
                    rect_upper = max(rect_list_upper) * unit
                    lef_dict[cell_name]['pin'][pin_name] = [rect_left, rect_lower, rect_right, rect_upper] # pin_rect
                    rect_list_left = []
                    rect_list_lower = []
                    rect_list_right = []
                    rect_list_upper = []

    return lef_dict


def read_lef_pin_map(path, lef_dic, unit):
    with open(path, 'r') as read_file:
        cell_name = ''
        pin_name = ''
        READ_MACRO = False

        for line in read_file:
            if line.lstrip().startswith('MACRO'):
                cell_name = line.split()[1]
                lef_dic[cell_name] = {}
                lef_dic[cell_name]['pin'] = {}
                READ_MACRO = True

            if READ_MACRO:

                if line.lstrip().startswith('SIZE'):
                    l = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
                    lef_dic[cell_name]['size'] = [unit * float(l[0]), unit * float(l[1])]

                elif line.lstrip().startswith('PIN') or line.lstrip().startswith('OBS'):
                    if line.lstrip().startswith('OBS'):
                        pin_name = 'OBS'
                    else:
                        pin_name = line.split()[1]
                    lef_dic[cell_name]['pin'][pin_name] = {}

                elif line.lstrip().startswith('LAYER'):
                    pin_layer = line.split()[1]
                    lef_dic[cell_name]['pin'][pin_name][pin_layer] = []

                elif line.lstrip().startswith('RECT'):
                    l = line.split()
                    lef_dic[cell_name]['pin'][pin_name][pin_layer].append([float(l[1])* unit,float(l[2])* unit,float(l[3])* unit,float(l[4])* unit])

    return lef_dic


class ReadInnovusOutput:
    def __init__(self, root_dir, arg, save_name, lef_dict, lef_dict_jnet=None):
        self.save_name = save_name
        self.save_path = arg.save_path
        self.unit = arg.unit
        
        # Path for LEF/DEF and reports from Innovus.
        self.lef_dict = lef_dict
        self.lef_dict_jnet = lef_dict_jnet
        self.root_dir = root_dir 
        self.place_def_path = os.path.join(root_dir, arg.place_def_name)
        self.route_def_path = os.path.join(root_dir, arg.route_def_name)
        self.eGR_congestion_path = os.path.join(root_dir, arg.eGR_congestion_name) 
        self.route_congestion_path = os.path.join(root_dir, arg.route_congestion_name)    
        self.drc_path = os.path.join(root_dir, arg.drc_rpt_name)
        self.twf_path = os.path.join(root_dir, arg.twf_rpt_name)
        self.power_path = os.path.join(root_dir, arg.power_rpt_name)
        self.ir_path = os.path.join(root_dir, arg.ir_rpt_name)
        
        # The LEF/DEF we released are scaled, and it can be reverted with the scaling factor (arg.scaling). 
        # Contact us with email to get the scaling factor if you need to revert the scaling.         
        if not arg.scaling:
            self.scaling = 1
        else:
            self.scaling = arg.scaling
        
        self.die_area = None                   # Total area of die (chip). Currently not used.
        self.gcell_size = [-1,-1]              # Number of Gcell grids.
        self.gcell_coordinate_x = []           # Coordinate of Gcell grids in DEF.
        self.gcell_coordinate_y = []
        # The geometric information of all instances(cells)/nets/pins from DEF.
        # We need to read both route DEF and place DEF as the instances may be changed due to optDesign.
        self.route_instance_dict = {}   # contains information of instances {inst_name:[std_cell_name, [inst_location_w/h], inst_direction]}
        self.route_net_dict = {}        # contains instance names and pin names in each nets {net_name:[pin_names]}
        self.route_pin_dict = {}        # contains information of IO pins {pin_name{'layer':layer, 'rect':rect, 'location':location, 'direction':direction}}
        self.place_instance_dict = {}   # same as route
        self.place_net_dict = {}
        self.place_pin_dict = {}

        # The placement(locations) of all instances(cells) from DEF
        self.instance_placement_dict = {}
        
        # Array of congestion for eGR (early global routing/trail routing) in 2 directions (horizontal/vertical) and 2 definitions (utilization/overflow of tracks).
        self.congestion_horizontal_util_eGR = []
        self.congestion_vertical_util_eGR = []
        self.congestion_horizontal_overflow_eGR = []
        self.congestion_vertical_overflow_eGR = []
        self.hotspot_eGR = 0
        # Array of congestion for GR (global routing) in 2 directions (horizontal/vertical) and 2 definitions (utilization/overflow of tracks).
        self.congestion_horizontal_util = []
        self.congestion_vertical_util = []
        self.congestion_horizontal_overflow = []
        self.congestion_vertical_overflow = []
        self.hotspot = 0

        # Array of routability features.
        self.cell_density=None
        self.RUDY = None
        self.RUDY_long = None
        self.RUDY_short = None
        self.pin_RUDY = None
        self.pin_RUDY_long = None
        
        # Dict for DRC with types as key and locations as value.
        self.drc_dict = {}
        
        # IR drop features.
        self.n_time_window = int(arg.n_time_window)
        self.tw_dict = {}
        self.power_dict = {}

        self.power_t = None
        self.power_i = None
        self.power_s = None
        self.power_sca = None
        self.power_all = None
        self.ir_map = None
        
    """
    Read DEF dumped after placement
    get macro_region (routability feature), place_instance_dict, place_net_dict and place_pin_dict.
    """

    def read_place_def(self):
        if is_gzip_file(self.place_def_path):
            read_file = gzip.open(self.place_def_path,"rt")
        else:
            read_file = open(self.place_def_path,"r")

        READ_MACROS = False
        READ_NETS = False
        READ_PINS = False
        macro_map = np.zeros(self.gcell_size)
        macro_map_with_halo = np.zeros(self.gcell_size)
        net = ''
        for line in read_file:
            line = line.lstrip()
            if line.startswith("DIEAREA"):
                die_coordinate = re.findall(r'\d+', line)
                self.die_area = (int(die_coordinate[2]), int(die_coordinate[3]))
            elif line.startswith("COMPONENTS"):
                READ_MACROS = True
            elif line.startswith("END COMPONENTS"):
                READ_MACROS = False
            elif line.startswith("NETS"):
                READ_NETS =True
            elif line.startswith("END NETS") or line.startswith("SPECIALNETS"):
                READ_NETS = False
            elif line.startswith('PIN'):
                READ_PINS =True
            elif line.startswith('END PINS'):
                READ_PINS = False

            if READ_MACROS:                                         # get macro_region (routability feature)
                if "FIXED" in line:
                    instance = line.split()
                    l = instance.index('(')
                    coordinate = (int(instance[l + 1]), int(instance[l + 2]))
                    self.place_instance_dict[instance[1]] = [instance[2], coordinate, instance[l+4]]
                    x_y_lower_left_gcell, x_y_upper_right_gcell, x_y_lower_left, x_y_upper_right = self.get_macro_region(line, coordinate, self.lef_dict[instance[2]]['size'])
                    macro_map[x_y_lower_left_gcell[0]:x_y_upper_right_gcell[0], x_y_lower_left_gcell[1]:x_y_upper_right_gcell[1]] = np.ones([x_y_upper_right_gcell[0] - x_y_lower_left_gcell[0], x_y_upper_right_gcell[1] - x_y_lower_left_gcell[1]])

                elif line.lstrip().startswith('+ HALO'):
                    l = line.split()
                    halo = [int(l[2]), int(l[3]), int(l[4]), int(l[5])]
                    x_y_lower_left = [x_y_lower_left[0] - halo[0], x_y_lower_left[1] - halo[1]]
                    x_y_upper_right = [x_y_upper_right[0] + halo[2], x_y_upper_right[1] + halo[3]]
                    x_y_lower_left_gcell[0] = bisect.bisect_left(self.gcell_coordinate_x, x_y_lower_left[0])    # map to gcell grids
                    x_y_lower_left_gcell[1] = bisect.bisect_left(self.gcell_coordinate_y, x_y_lower_left[1])   
                    x_y_upper_right_gcell[0] = bisect.bisect_left(self.gcell_coordinate_x, x_y_upper_right[0])
                    x_y_upper_right_gcell[1] = bisect.bisect_left(self.gcell_coordinate_y, x_y_upper_right[1])
                    macro_map_with_halo[x_y_lower_left_gcell[0]:x_y_upper_right_gcell[0], x_y_lower_left_gcell[1]:x_y_upper_right_gcell[1]] = np.ones([x_y_upper_right_gcell[0] - x_y_lower_left_gcell[0], x_y_upper_right_gcell[1] - x_y_lower_left_gcell[1]])
                elif "PLACED" in line:                              # get place_instance_dict             
                    instance = line.split()
                    l = instance.index('(')
                    if instance[1]=='(':
                        print(line)
                    self.place_instance_dict[instance[1]] = [instance[2], (int(instance[l+1]), int(instance[l+2])), instance[l+4]]
            if READ_NETS:                                           # get route_net_dict
                if line.startswith('-'):
                    net = line.split()[1]
                    self.place_net_dict[net] = []
                elif line.startswith('('):                          # get pin names in each net
                    l = line.split()
                    n = 0
                    for k in l:
                        if k == '(':
                            self.place_net_dict[net].append([l[n+1], l[n+2]])
                        n += 1
            if READ_PINS:                                           # get place_pin_dict (for primary IO pins)
                if line.startswith('-'):
                    pin = line.split()[1]
                elif line.strip().startswith('+ LAYER'):
                    pin_rect = re.findall(r'\d+', line)
                    self.place_pin_dict[pin] = [int(pin_rect[-4]), int(pin_rect[-3]), int(pin_rect[-2]), int(pin_rect[-1])]
        self.macro_map = macro_map
        self.macro_map_with_halo = macro_map_with_halo
        save(self.save_path, 'macro_region', self.save_name, self.macro_map_with_halo) # save macro_region feature
        read_file.close()

    def get_macro_region(self, line, x_y, area): # function used by read_place_def
        x_y_lower_left = [int(x_y[0]), int(x_y[1])]
        x_y_lower_left_gcell = []
        x_y_upper_right_gcell = []
        x_y_lower_left_gcell.append(bisect.bisect_left(self.gcell_coordinate_x, x_y_lower_left[0]))
        x_y_lower_left_gcell.append(bisect.bisect_left(self.gcell_coordinate_y, x_y_lower_left[1]))
        direction = instance_direction_rect(line)
        x_y_upper_right = [x_y_lower_left[0] + area[0] * direction[0] + area[1] * direction[1], x_y_lower_left[1] + area[0] * direction[2] + area[1] * direction[3]]
        x_y_upper_right_gcell.append(bisect.bisect_left(self.gcell_coordinate_x, x_y_upper_right[0]))
        x_y_upper_right_gcell.append(bisect.bisect_left(self.gcell_coordinate_y, x_y_upper_right[1]))
        return x_y_lower_left_gcell, x_y_upper_right_gcell, x_y_lower_left, x_y_upper_right

    """
    Read DEF dumped after global routing (after detailed routing recommended), 
    get gcell_coordinate, route_instance_dict, route_net_dict and route_pin_dict.
    """

    def read_route_def(self):
        self.gcell_size= [-1,-1]
        self.gcell_coordinate_x = []
        self.gcell_coordinate_y = []
        GCELLX = []
        GCELLY = []
        if is_gzip_file(self.route_def_path):
            read_file = gzip.open(self.route_def_path,"rt")
        else:
            read_file = open(self.route_def_path,"r")

        READ_GCELL = False
        READ_MACROS = False
        READ_NETS = False
        READ_PINS = False
        net = ''
        for line in read_file:
            line = line.lstrip()
            if line.startswith("COMPONENTS"):
                READ_MACROS = True
            elif line.startswith("END COMPONENTS"):
                READ_MACROS = False
            elif line.startswith("NETS"):
                READ_NETS =True
            elif line.startswith("END NETS") or line.startswith("SPECIALNETS"):
                READ_NETS = False
            elif line.startswith('PIN'):
                READ_PINS =True
            elif line.startswith('END PINS'):
                READ_PINS = False
            elif line.startswith("GCELLGRID"):
                READ_GCELL = True
            elif line.startswith("VIAS"):
                READ_GCELL = False
                if len(GCELLX) <= 2:
                    raise ValueError
                if int(GCELLX[0][0]) < int(GCELLX[-1][0]):
                    GCELLX.reverse()
                    GCELLY.reverse()

                top = GCELLY.pop()
                for i in range(top[1]-1):
                    self.gcell_coordinate_y.append(top[0]+(i+1)*top[2])
                for i in range(len(GCELLY)):
                    top = GCELLY.pop()
                    for i in range(top[1]):
                        self.gcell_coordinate_y.append(self.gcell_coordinate_y[-1]+top[2])
                self.gcell_coordinate_y = np.array(self.gcell_coordinate_y)

                top = GCELLX.pop()
                for i in range(top[1]-1):
                    self.gcell_coordinate_x.append(top[0]+(i+1)*top[2])
                for i in range(len(GCELLX)):
                    top = GCELLX.pop()
                    for i in range(top[1]):
                        self.gcell_coordinate_x.append(self.gcell_coordinate_x[-1]+top[2])
                self.gcell_coordinate_x = np.array(self.gcell_coordinate_x)

            if READ_GCELL:   # get gcell_coordinate
                instance = line.split()
                if not len(instance) == 8:
                    continue
                else:
                    gcell = [int(int(instance[2])),int(instance[4]),int(int(instance[6]))]  # at x do y step z
                    if 'Y' in line:
                        self.gcell_size[1] += int(instance[4])
                        GCELLY.append(gcell)

                    elif 'X' in line:
                        self.gcell_size[0] += int(instance[4])
                        GCELLX.append(gcell)

            if READ_MACROS :                                            # get route_instance_dict
                if "FIXED" in line or "PLACED" in line:
                    instance = line.split()
                    l = instance.index('(')
                    self.route_instance_dict[instance[1].replace('\\', '')] = [instance[2], (int(instance[l+1]), int(instance[l+2])), instance[l+4]]

            if READ_NETS:
                if line.startswith('-'):
                    net = line.split()[1].replace('\\', '')             # get route_net_dict
                    self.route_net_dict[net] = []

                elif line.startswith('('):                              # get pin names in each net
                    l = line.split()
                    n = 0
                    for k in l:
                        if k == '(':
                            self.route_net_dict[net].append(l[n+1].replace('\\', ''))
                        n += 1
            if READ_PINS:                                               # get route_pin_dict (for primary IO pins)
                if line.startswith('-'):
                    pin = line.split()[1]
                elif line.strip().startswith('+ LAYER'):
                    pin_rect = re.findall(r'\d+', line)
                    self.route_pin_dict[pin] = {}
                    self.route_pin_dict[pin]['layer'] = line.split()[2]
                    self.route_pin_dict[pin]['rect'] = [int(pin_rect[-4]), int(pin_rect[-3]), int(pin_rect[-2]), int(pin_rect[-1])]
                elif line.strip().startswith('+ PLACED'):
                    data = line.split()
                    self.route_pin_dict[pin]['location'] = [int(data[3]), int(data[4])]
                    self.route_pin_dict[pin]['direction'] = data[6]
        read_file.close()
    
    """
    Read early global routing congestion report, save eGR congestion map (routability feature).
    """

    def read_eGR_congestion(self):
        with open(self.eGR_congestion_path, 'r') as read_file:
            first = True
            for line in read_file:
                if line.startswith('('):
                    if first:
                        first_line = line
                        first = False
                    resources = re.findall(r'-?\d+', line)

                    capacity_vertical = float(resources[-3])
                    overflow_vertical = float(resources[-4])
                    capacity_horizontal = float(resources[-1])
                    overflow_horizontal = float(resources[-2])

                    if int(capacity_vertical) == 0:
                        self.congestion_vertical_util_eGR.append(0)
                        self.congestion_vertical_overflow_eGR.append(0)
                    elif int(overflow_vertical) < 0 :
                        self.congestion_vertical_util_eGR.append(1)
                        self.congestion_vertical_overflow_eGR.append(abs(overflow_vertical/capacity_vertical))
                        self.hotspot_eGR += 1
                    else:
                        self.congestion_vertical_util_eGR.append(1 - overflow_vertical/capacity_vertical)
                        self.congestion_vertical_overflow_eGR.append(0)
                    if int(capacity_horizontal) == 0:
                        self.congestion_horizontal_util_eGR.append(0)
                        self.congestion_horizontal_overflow_eGR.append(0)
                    elif int(overflow_horizontal) < 0:
                        self.congestion_horizontal_util_eGR.append(1)
                        self.congestion_horizontal_overflow_eGR.append(abs(overflow_horizontal/overflow_horizontal))
                        self.hotspot_eGR += 1
                    else:
                        self.congestion_horizontal_util_eGR.append(1 - overflow_horizontal / capacity_horizontal)
                        self.congestion_horizontal_overflow_eGR.append(0)
        first_resources = re.findall(r'\d+', first_line)
        gcell_size_eGR = [int((int(resources[0]) - int(first_resources[0])) / (int(first_resources[2]) - int(first_resources[0]))) + 1, 
        int((int(resources[-5]) - int(first_resources[-5])) / (int(first_resources[3]) - int(first_resources[1]))) + 1]
        self.congestion_horizontal_util_eGR = np.array(self.congestion_horizontal_util_eGR).reshape((gcell_size_eGR[0], gcell_size_eGR[1]), order='F')
        self.congestion_horizontal_overflow_eGR = np.array(self.congestion_horizontal_overflow_eGR).reshape((gcell_size_eGR[0], gcell_size_eGR[1]), order='F')
        self.congestion_vertical_util_eGR = np.array(self.congestion_vertical_util_eGR).reshape((gcell_size_eGR[0], gcell_size_eGR[1]), order='F')
        self.congestion_vertical_overflow_eGR = np.array(self.congestion_vertical_overflow_eGR).reshape((gcell_size_eGR[0], gcell_size_eGR[1]), order='F')
        
        save(self.save_path, 'congestion/congestion_early_global_routing/utilization_based/congestion_eGR_horizontal_util', self.save_name, self.congestion_horizontal_util_eGR)
        save(self.save_path, 'congestion/congestion_early_global_routing/overflow_based/congestion_eGR_horizontal_overflow', self.save_name, self.congestion_horizontal_overflow_eGR)
        save(self.save_path, 'congestion/congestion_early_global_routing/utilization_based/congestion_eGR_vertical_util', self.save_name, self.congestion_vertical_util_eGR)
        save(self.save_path, 'congestion/congestion_early_global_routing/overflow_based/congestion_eGR_vertical_overflow', self.save_name, self.congestion_vertical_overflow_eGR)

    """
    Read global routing congestion report, save congestion map (routability feature).
    """
    
    def read_route_congestion(self):
        with open(self.route_congestion_path, 'r') as read_file:
            for line in read_file:
                if line.startswith('('):
                    resources = re.findall(r'-?\d+', line)

                    capacity_vertical = float(resources[-4])
                    used_vertical = float(resources[-5])
                    overflow_vertical = float(resources[-6])
                    capacity_horizontal = float(resources[-1])
                    used_horizontal = float(resources[-2])
                    overflow_horizontal = float(resources[-3])

                    if int(capacity_vertical) == 0:
                        self.congestion_vertical_util.append(0)
                        self.congestion_vertical_overflow.append(0)
                    elif int(overflow_vertical) < 0:
                        self.congestion_vertical_util.append(1)
                        self.congestion_vertical_overflow.append(- overflow_vertical / capacity_vertical)
                        self.hotspot += 1
                    else:
                        self.congestion_vertical_util.append(1 - (used_vertical / capacity_vertical))
                        self.congestion_vertical_overflow.append(0)
                    if int(capacity_horizontal) == 0:
                        self.congestion_horizontal_util.append(0)
                        self.congestion_horizontal_overflow.append(0)
                    elif int(overflow_horizontal) < 0:
                        self.congestion_horizontal_util.append(1)
                        self.congestion_horizontal_overflow.append(- overflow_horizontal / capacity_horizontal)
                        self.hotspot += 1
                    else:
                        self.congestion_horizontal_util.append(1 - (used_horizontal / capacity_horizontal))
                        self.congestion_horizontal_overflow.append(0)
                    self.gcell_coordinate_x.append(int(resources[2]))
                    self.gcell_coordinate_y.append(int(resources[3]))

        self.congestion_horizontal_util = np.array(self.congestion_horizontal_util).reshape((self.gcell_size[0], self.gcell_size[1]), order='F')
        self.congestion_horizontal_overflow = np.array(self.congestion_horizontal_overflow).reshape((self.gcell_size[0], self.gcell_size[1]), order='F')
        self.congestion_vertical_util = np.array(self.congestion_vertical_util).reshape((self.gcell_size[0], self.gcell_size[1]), order='F')
        self.congestion_vertical_overflow = np.array(self.congestion_vertical_overflow).reshape((self.gcell_size[0], self.gcell_size[1]), order='F')
        self.gcell_coordinate_x = np.array(self.gcell_coordinate_x).reshape((self.gcell_size[0], self.gcell_size[1]), order='F')[..., 0]
        self.gcell_coordinate_y = np.array(self.gcell_coordinate_y).reshape((self.gcell_size[0], self.gcell_size[1]), order='F')[0, ...]
        assert self.gcell_coordinate_x.size == self.gcell_size[0] and self.gcell_coordinate_y.size == self.gcell_size[1]

        save(self.save_path, 'congestion/congestion_global_routing/utilization_based/congestion_GR_horizontal_util', self.save_name, self.congestion_horizontal_util)
        save(self.save_path, 'congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow', self.save_name, self.congestion_horizontal_overflow)
        save(self.save_path, 'congestion/congestion_global_routing/utilization_based/congestion_GR_vertical_util', self.save_name, self.congestion_vertical_util)
        save(self.save_path, 'congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow', self.save_name, self.congestion_vertical_overflow)

    """
    Read global routing congestion report, report total overflow and average overflow.
    """
    
    def read_congestion_overflow(self):
        with open(self.route_congestion_path, 'r') as read_file:
            count = 0 
            HC = 0
            VC = 0
            for line in read_file:
                if line.startswith('('):
                    resources = re.findall(r'-?\d+', line)

                    overflow_vertical = float(resources[-6])
                    VC += overflow_vertical
                    overflow_horizontal = float(resources[-3])
                    HC += overflow_horizontal
                    count +=1
            print(HC,VC, HC/float(count), VC/float(count))

    '''
    Read place DEF, get instance placement in micron or in GCell grids (graph features).
    '''
    
    def read_instance_placement(self):
        out_instance_dic_micron = {}
        out_instance_dic_gcell = {}
        with open(self.route_def_path, 'r') as read_file:
            READ_MACROS = False
            for line in read_file:
                if "COMPONENTS" in line:                                           
                    READ_MACROS = True
                    if "END COMPONENTS" in line:
                        READ_MACROS = False

                if READ_MACROS :
                    if "FIXED" in line or "PLACED" in line:
                        line = line.split()
                        left = line.index('(')
                        instance_name = line[1].replace('\\', '') 
                        instance_direction = instance_direction_rect(line[left+4])

                        instance_size = self.lef_dict[line[2]]['size']
                        instance_coord = [float(line[left+1]), float(line[left+2]), 
                        float(int(line[left+1]) + instance_size[0] * instance_direction[0] + instance_size[1] * instance_direction[1]), 
                        float(int(line[left+2]) + instance_size[0] * instance_direction[2] + instance_size[1] * instance_direction[3])]

                        out_instance_dic_micron[instance_name] = [x*self.scaling/self.unit for x in instance_coord]

                        instance_coord_gcell = [bisect.bisect_left(self.gcell_coordinate_x, instance_coord[0]),
                        bisect.bisect_left(self.gcell_coordinate_y, instance_coord[1]),
                        bisect.bisect_left(self.gcell_coordinate_x, instance_coord[2]),
                        bisect.bisect_left(self.gcell_coordinate_y, instance_coord[3])]
                        if instance_coord_gcell[0]>=len(self.gcell_coordinate_x):
                            instance_coord_gcell[0]=len(self.gcell_coordinate_x)-1
                        if instance_coord_gcell[1]>=len(self.gcell_coordinate_y):
                            instance_coord_gcell[1]=len(self.gcell_coordinate_y)-1
                        if instance_coord_gcell[2]>=len(self.gcell_coordinate_x):
                            instance_coord_gcell[2]=len(self.gcell_coordinate_x)-1
                        if instance_coord_gcell[3]>=len(self.gcell_coordinate_y):
                            instance_coord_gcell[3]=len(self.gcell_coordinate_y)-1

                        out_instance_dic_gcell[instance_name] = instance_coord_gcell

        save(self.save_path, 'instance_placement_micron', self.save_name, out_instance_dic_micron)
        save(self.save_path, 'instance_placement_gcell', self.save_name, out_instance_dic_gcell)

    '''
    Read route DEF, get pin positions (graph features).
    '''
    
    def read_route_pin_position(self):
        if not self.route_instance_dict: # read route DEF if not read before.
            self.gcell_size= [-1,-1]
            self.gcell_coordinate_x = []
            self.gcell_coordinate_y = []
            GCELLX = []
            GCELLY = []
            with open(self.route_def_path, 'r') as read_file:
                READ_GCELL = False
                READ_MACROS = False
                READ_NETS = False
                READ_PINS = False
                for line in read_file:
                    line = line.lstrip()
                    if line.startswith("DIEAREA"):
                        die_coordinate = re.findall(r'\d+', line)
                        self.die_area = (int(die_coordinate[2]), int(die_coordinate[3]))
                    if line.startswith("COMPONENTS"):
                        READ_MACROS = True
                    elif line.startswith("END COMPONENTS"):
                        READ_MACROS = False
                    elif line.startswith("NETS"):
                        READ_NETS =True
                    elif line.startswith("END NETS"):
                        READ_NETS = False
                    elif line.startswith('PIN'):
                        READ_PINS =True
                    elif line.startswith('END PINS'):
                        READ_PINS = False
                    elif line.startswith("GCELLGRID"):
                        READ_GCELL = True
                    elif line.startswith("VIAS"):
                        READ_GCELL = False
                        if len(GCELLX) <= 2:
                            raise ValueError
                        if int(GCELLX[0][0]) < int(GCELLX[-1][0]):
                            GCELLX.reverse()
                            GCELLY.reverse()

                        top = GCELLY.pop()
                        for i in range(top[1]-1):
                            self.gcell_coordinate_y.append(top[0]+(i+1)*top[2])
                        for i in range(len(GCELLY)):
                            top = GCELLY.pop()
                            for i in range(top[1]):
                                self.gcell_coordinate_y.append(self.gcell_coordinate_y[-1]+top[2])
                        self.gcell_coordinate_y.pop()
                        self.gcell_coordinate_y = np.array(self.gcell_coordinate_y)


                        top = GCELLX.pop()
                        for i in range(top[1]-1):
                            self.gcell_coordinate_x.append(top[0]+(i+1)*top[2])
                        for i in range(len(GCELLX)):
                            top = GCELLX.pop()
                            for i in range(top[1]):
                                self.gcell_coordinate_x.append(self.gcell_coordinate_x[-1]+top[2])
                        self.gcell_coordinate_x.pop()
                        self.gcell_coordinate_x = np.array(self.gcell_coordinate_x)


                    if READ_GCELL:   # get gcell_coordinate
                        instance = line.split()
                        if not len(instance) == 8:
                            continue
                        else:
                            gcell = [int(int(instance[2])),int(instance[4]),int(int(instance[6]))]  # at x do y step z
                            if 'Y' in line:
                                self.gcell_size[1] += int(instance[4])
                                GCELLY.append(gcell)

                            elif 'X' in line:
                                self.gcell_size[0] += int(instance[4])
                                GCELLX.append(gcell)

                    if READ_MACROS :
                        if "FIXED" in line or "PLACED" in line:
                            instance = line.split()

                            l = instance.index('(')
                            self.route_instance_dict[instance[1].replace('\\', '')] = [instance[2], (int(instance[l+1]), int(instance[l+2])), instance[l+4]]        #key: inst's name; value: 1. name of std cell 2. location of inst 3. direction of inst

                    if READ_PINS:
                        if line.startswith('-'):
                            pin = line.split()[1]
                        elif line.strip().startswith('+ LAYER'):
                            pin_rect = re.findall(r'\d+', line)
                            self.route_pin_dict[pin] = {}
                            self.route_pin_dict[pin]['layer'] = line.split()[2]
                            self.route_pin_dict[pin]['rect'] = [int(pin_rect[-4]), int(pin_rect[-3]), int(pin_rect[-2]), int(pin_rect[-1])]
                        elif line.strip().startswith('+ PLACED'):
                            data = line.split()
                            self.route_pin_dict[pin]['location'] = [int(data[3]), int(data[4])]
                            self.route_pin_dict[pin]['direction'] = data[6]
                    if READ_NETS:
                        continue
        route_instance_dict = self.route_instance_dict

        pin_position_dict = {}
        for cell_name, cell_features in route_instance_dict.items():

            instance_location_on_chip = [0, 0]
            for pin_name, pin_coordinate in self.lef_dict[cell_features[0]]['pin'].items():           #pin_coordinate left/lower/right/upper
                direction = instance_direction_bottom_left(cell_features[2])
                std_cell_x, std_cell_y = self.lef_dict[cell_features[0]]['size']
                instance_location_on_chip[0] = cell_features[1][0] + direction[8] * std_cell_x + direction[9] * std_cell_y   
                instance_location_on_chip[1] = cell_features[1][1] + direction[10] * std_cell_x + direction[11] * std_cell_y
                
                pin_left = instance_location_on_chip[0] + pin_coordinate[0] * direction[4] + pin_coordinate[1] * direction[5] + pin_coordinate[2] * direction[0] + pin_coordinate[3] * direction[1]
                pin_lower = instance_location_on_chip[1] + pin_coordinate[0] * direction[6] + pin_coordinate[1] * direction[7] + pin_coordinate[2] * direction[2] + pin_coordinate[3] * direction[3]
                pin_right = instance_location_on_chip[0] + pin_coordinate[2] * direction[4] + pin_coordinate[3] * direction[5] + pin_coordinate[0] * direction[0] + pin_coordinate[1] * direction[1]
                pin_upper = instance_location_on_chip[1] + pin_coordinate[2] * direction[6] + pin_coordinate[3] * direction[7] + pin_coordinate[0] * direction[2] + pin_coordinate[1] * direction[3]
                pin_position_micron = [round(pin_left*self.scaling/self.unit,2), round(pin_lower*self.scaling/self.unit,2),  round(pin_right*self.scaling/self.unit,2), round(pin_upper*self.scaling/self.unit,2)]
                pin_position_dict['%s/%s'%(cell_name,pin_name)] = pin_position_micron
                
                pin_gcell_left = bisect.bisect_left(self.gcell_coordinate_x, pin_left)
                pin_gcell_right = bisect.bisect_left(self.gcell_coordinate_x, pin_right)
                pin_gcell_lower = bisect.bisect_left(self.gcell_coordinate_y, pin_lower)
                pin_gcell_upper = bisect.bisect_left(self.gcell_coordinate_y, pin_upper)
                if pin_gcell_left>=len(self.gcell_coordinate_x):
                    pin_gcell_left=len(self.gcell_coordinate_x)-1                
                if pin_gcell_right>=len(self.gcell_coordinate_x):
                    pin_gcell_right=len(self.gcell_coordinate_x)-1
                if pin_gcell_lower>=len(self.gcell_coordinate_y):
                    pin_gcell_lower=len(self.gcell_coordinate_y)-1
                if pin_gcell_upper>=len(self.gcell_coordinate_y):
                    pin_gcell_upper=len(self.gcell_coordinate_y)-1
                pin_position_gcell = [pin_gcell_left, pin_gcell_lower, pin_gcell_right, pin_gcell_upper]
                pin_position_dict['%s/%s'%(cell_name,pin_name)].extend(pin_position_gcell)
                
        for pin_name, pin_features in self.route_pin_dict.items():
            direction = instance_direction_bottom_left(pin_features['direction'])
            pin_location_on_chip = pin_features['location']
            pin_coordinate = pin_features['rect']
            pin_left = pin_location_on_chip[0] + pin_coordinate[0] * direction[4] + pin_coordinate[1] * direction[5] + pin_coordinate[2] * direction[0] + pin_coordinate[3] * direction[1]
            pin_lower = pin_location_on_chip[1] + pin_coordinate[0] * direction[6] + pin_coordinate[1] * direction[7] + pin_coordinate[2] * direction[2] + pin_coordinate[3] * direction[3]
            pin_right = pin_location_on_chip[0] + pin_coordinate[2] * direction[4] + pin_coordinate[3] * direction[5] + pin_coordinate[0] * direction[0] + pin_coordinate[1] * direction[1]
            pin_upper = pin_location_on_chip[1] + pin_coordinate[2] * direction[6] + pin_coordinate[3] * direction[7] + pin_coordinate[0] * direction[2] + pin_coordinate[1] * direction[3]
            pin_position_micron = [round(pin_left*self.scaling/self.unit,2), round(pin_lower*self.scaling/self.unit,2),  round(pin_right*self.scaling/self.unit,2), round(pin_upper*self.scaling/self.unit,2)]
            pin_position_dict[pin_name] = pin_position_micron
            
            pin_gcell_left = bisect.bisect_left(self.gcell_coordinate_x, pin_left)
            pin_gcell_right = bisect.bisect_left(self.gcell_coordinate_x, pin_right)
            pin_gcell_lower = bisect.bisect_left(self.gcell_coordinate_y, pin_lower)
            pin_gcell_upper = bisect.bisect_left(self.gcell_coordinate_y, pin_upper)
            if pin_gcell_left>=len(self.gcell_coordinate_x):
                pin_gcell_left=len(self.gcell_coordinate_x)-1                
            if pin_gcell_right>=len(self.gcell_coordinate_x):
                pin_gcell_right=len(self.gcell_coordinate_x)-1
            if pin_gcell_lower>=len(self.gcell_coordinate_y):
                pin_gcell_lower=len(self.gcell_coordinate_y)-1
            if pin_gcell_upper>=len(self.gcell_coordinate_y):
                pin_gcell_upper=len(self.gcell_coordinate_y)-1
            pin_position_gcell = [pin_gcell_left, pin_gcell_lower, pin_gcell_right, pin_gcell_upper]
            pin_position_dict[pin_name].extend(pin_position_gcell)
            
        save_path = os.path.join(self.save_path, 'pin_positions', self.save_name)
        os.system("mkdir -p %s " % (os.path.dirname(save_path)))
        np.savez_compressed(save_path, pin_positions = pin_position_dict)

    """
    Functions for computing density.
    """
    def compute_density(self, density, location_on_gcell):
        gcell_left, gcell_lower, gcell_right, gcell_upper = location_on_gcell
        if gcell_left == self.gcell_coordinate_x.size:  # avoid getting size+1 from bisect_left
            gcell_left = gcell_left - 1
        if gcell_lower == self.gcell_coordinate_y.size:
            gcell_lower = gcell_lower - 1
        if gcell_right == self.gcell_coordinate_x.size:
            gcell_right = gcell_right - 1
        if gcell_upper == self.gcell_coordinate_y.size:
            gcell_upper = gcell_upper - 1

        for j in my_range(gcell_left, gcell_right):
            for k in my_range(gcell_lower, gcell_upper):
                density[j, k] += 1
        return density

    def compute_density_with_overlap(self, density, location_on_coordinate, location_on_gcell):
        x_left, y_lower, x_right, y_upper = location_on_coordinate            
        gcell_left, gcell_lower, gcell_right, gcell_upper = location_on_gcell 
        if gcell_left == self.gcell_coordinate_x.size:  
            gcell_left = gcell_left - 1
        if gcell_lower == self.gcell_coordinate_y.size:
            gcell_lower = gcell_lower - 1
        if gcell_right == self.gcell_coordinate_x.size:
            gcell_right = gcell_right - 1
        if gcell_upper == self.gcell_coordinate_y.size:
            gcell_upper = gcell_upper - 1

        if x_left == self.gcell_coordinate_x[gcell_left-1]:  
            gcell_left = gcell_right
        if y_lower == self.gcell_coordinate_y[gcell_lower-1]:
            gcell_lower = gcell_upper
        for j in my_range(gcell_left, gcell_right):
            for k in my_range(gcell_lower, gcell_upper):
                if j == 0:
                    left = -10                             # the border of die is -10
                else:
                    left = self.gcell_coordinate_x[j - 1]
                if k == 0:
                    lower = -10 
                else:
                    lower = self.gcell_coordinate_y[k - 1]       
                right = self.gcell_coordinate_x[j]
                upper = self.gcell_coordinate_y[k]
                if (right - left) * (upper - lower) == 0:
                 print(right, left)
                overlap = ((min(right, x_right) - max(left, x_left)) * (min(upper, y_upper) - max(lower, y_lower))) / ((right - left) * (upper - lower))
                density[j,k] += overlap
        return density

    """
    Get cell_density (routability feature), request running read_place_def first. 
    """
    
    def compute_cell_density(self):
        cell_density = np.zeros(self.gcell_size)
        for n in self.place_instance_dict:
            instance = self.place_instance_dict[n]
            cell_x_left_gcell = bisect.bisect_left(self.gcell_coordinate_x, instance[1][0])
            cell_y_lower_gcell = bisect.bisect_left(self.gcell_coordinate_y, instance[1][1])
            cell_size = self.lef_dict[instance[0]]['size']
            direction = instance_direction_rect(instance[2])
            cell_x_right = instance[1][0] + cell_size[0] * direction[0] + cell_size[1] * direction[1]
            cell_y_upper = instance[1][1] + cell_size[0] * direction[2] + cell_size[1] * direction[3]
            cell_x_right_gcell = bisect.bisect_left(self.gcell_coordinate_x, cell_x_right)
            cell_y_upper_gcell = bisect.bisect_left(self.gcell_coordinate_y, cell_y_upper)
            cell_coordinate = [cell_x_left_gcell, cell_y_lower_gcell, cell_x_right_gcell, cell_y_upper_gcell]
            cell_density = self.compute_density(cell_density, cell_coordinate)
        self.cell_density = cell_density
        save(self.save_path, 'cell_density', self.save_name, self.cell_density)

    """
    Get RUDY related features (routability feature).
    """
    
    def get_RUDY(self):
        RUDY = np.zeros((self.gcell_size[0], self.gcell_size[1]))
        RUDY_long = np.zeros((self.gcell_size[0], self.gcell_size[1]))
        RUDY_short = np.zeros((self.gcell_size[0], self.gcell_size[1]))
        pin_RUDY = np.zeros((self.gcell_size[0], self.gcell_size[1]))
        pin_RUDY_long = np.zeros((self.gcell_size[0], self.gcell_size[1]))
        for net in self.place_net_dict.keys():
            pin_location_on_chip_x = []
            pin_location_on_chip_y = []
            pin_location_on_gcell = []
            if len(self.place_net_dict[net])==1:
                continue
            for cell_pin_pair in self.place_net_dict[net]:     #e.g. axi_interconnect_i/axi_node_i/_RESP_BLOCK_GEN\[1\].RESP_BLOCK/BR_ALLOC/U70 I cell pin pair
                instance_location_on_chip = [0, 0]

                if cell_pin_pair[0] == 'PIN':
                    pin_left, pin_lower, pin_right, pin_upper = self.place_pin_dict[cell_pin_pair[1]]
                else:
                    cell_name = cell_pin_pair[0]
                    pin_name = cell_pin_pair[1]
                    direction = instance_direction_bottom_left(self.place_instance_dict[cell_name][2])
                    std_cell_x, std_cell_y = self.lef_dict[self.place_instance_dict[cell_name][0]]['size']
                    # convert coordinate to lower left corner
                    instance_location_on_chip[0] = self.place_instance_dict[cell_name][1][0] + direction[8] * std_cell_x + direction[9] * std_cell_y
                    instance_location_on_chip[1] = self.place_instance_dict[cell_name][1][1] + direction[10] * std_cell_x + direction[11] * std_cell_y
                    
                    #pin_location_on_instance left/lower/right/upper
                    pin_location_on_instance = self.lef_dict[self.place_instance_dict[cell_name][0]]['pin'][pin_name]
                    pin_left = instance_location_on_chip[0] + pin_location_on_instance[0] * direction[4] + pin_location_on_instance[1] * direction[5] + pin_location_on_instance[2] * direction[0] + pin_location_on_instance[3] * direction[1]
                    pin_lower = instance_location_on_chip[1] + pin_location_on_instance[0] * direction[6] + pin_location_on_instance[1] * direction[7] + pin_location_on_instance[2] * direction[2] + pin_location_on_instance[3] * direction[3]
                    pin_right = instance_location_on_chip[0] + pin_location_on_instance[2] * direction[4] + pin_location_on_instance[3] * direction[5] + pin_location_on_instance[0] * direction[0] + pin_location_on_instance[1] * direction[1]
                    pin_upper = instance_location_on_chip[1] + pin_location_on_instance[2] * direction[6] + pin_location_on_instance[3] * direction[7] + pin_location_on_instance[0] * direction[2] + pin_location_on_instance[1] * direction[3]
                pin_gcell_left = bisect.bisect_left(self.gcell_coordinate_x, pin_left)
                pin_gcell_right = bisect.bisect_left(self.gcell_coordinate_x, pin_right)
                pin_gcell_lower = bisect.bisect_left(self.gcell_coordinate_y, pin_lower)
                pin_gcell_upper = bisect.bisect_left(self.gcell_coordinate_y, pin_upper)
                pin_location_on_gcell.append([pin_gcell_left, pin_gcell_lower, pin_gcell_right, pin_gcell_upper])


                pin_location_on_chip_x.append(pin_left)   # collect all the pin of the net to get net bounding box
                pin_location_on_chip_x.append(pin_right)
                pin_location_on_chip_y.append(pin_lower)
                pin_location_on_chip_y.append(pin_upper)
            if pin_location_on_chip_x == []:
                continue
            else:
                net_left = min(pin_location_on_chip_x)
                net_right = max(pin_location_on_chip_x)
                net_lower = min(pin_location_on_chip_y)
                net_upper = max(pin_location_on_chip_y)
                net_gcell_left = bisect.bisect_left(self.gcell_coordinate_x, net_left)
                net_gcell_right = bisect.bisect_left(self.gcell_coordinate_x, net_right)
                net_gcell_lower = bisect.bisect_left(self.gcell_coordinate_y, net_lower)
                net_gcell_upper = bisect.bisect_left(self.gcell_coordinate_y, net_upper)
                temp_rudy = np.zeros(self.gcell_size)
                location_on_coordinate = [net_left, net_lower, net_right, net_upper]
                location_on_gcell = [net_gcell_left, net_gcell_lower, net_gcell_right, net_gcell_upper]
                temp_rudy, long_or_short = self.compute_RUDY(temp_rudy, location_on_coordinate, location_on_gcell)
                RUDY += temp_rudy
                pin_RUDY_weight = 1/(net_right - net_left) + 1/(net_upper - net_lower)
                if long_or_short == 'long':                                    # short: cover only 1 Gcell, long: cover more than 1
                    RUDY_long += temp_rudy
                elif long_or_short == 'short':
                    RUDY_short += temp_rudy
                else:
                    raise ValueError('rudy computation error')

                for pin in pin_location_on_gcell:
                    temp_pin_rudy = np.zeros(self.gcell_size)
                    temp_pin_rudy = self.compute_density(temp_pin_rudy, pin)
                    pin_RUDY += temp_pin_rudy * pin_RUDY_weight
                    if long_or_short == 'long':
                        pin_RUDY_long += temp_pin_rudy * pin_RUDY_weight

        self.RUDY = RUDY
        self.RUDY_long = RUDY_long
        self.RUDY_short = RUDY_short
        self.pin_RUDY = pin_RUDY
        self.pin_RUDY_long = pin_RUDY_long
        save(self.save_path, 'RUDY/RUDY', self.save_name, self.RUDY)
        save(self.save_path, 'RUDY/RUDY_long', self.save_name, self.RUDY_long)
        save(self.save_path, 'RUDY/RUDY_short', self.save_name, self.RUDY_short)
        save(self.save_path, 'RUDY/RUDY_pin', self.save_name, self.pin_RUDY)
        save(self.save_path, 'RUDY/RUDY_pin_long', self.save_name, self.pin_RUDY_long)

    """
    Function used by get_RUDY.
    """

    def compute_RUDY(self, rudy, location_on_coordinate, location_on_gcell):
        long_or_short = ''
        x_left, y_lower, x_right, y_upper = location_on_coordinate 
        gcell_left, gcell_lower, gcell_right, gcell_upper = location_on_gcell
        w = (x_right - x_left)
        h = (y_upper - y_lower)
        weight = 1/w + 1/h

        if not gcell_right > gcell_left and not gcell_upper > gcell_lower:
            long_or_short = 'short'
        else:
            long_or_short = 'long'

        rudy = weight * self.compute_density_with_overlap(rudy, location_on_coordinate, location_on_gcell)
        
        return rudy, long_or_short

    """
    Get DRC related features (routability feature).
    """
    
    def get_DRC(self):
        self.read_DRC()
        self.compute_DRC_density()
        
    """
    Function used by get_DRC. Read DRC report and extract DRC rects.
    """

    def read_DRC(self):
        with open(self.drc_path, 'r') as read_file:
            for line in read_file:
                if "Net" in line or "Cell" in line:
                    drc_data = re.findall(r'[(](.*?)[)]', line)   # 0 type 1 layer
                    drc_type = drc_data[0].strip().replace('\t', '')
                    drc_layer = drc_data[1].strip()
                    if not drc_type in self.drc_dict:
                        self.drc_dict[drc_type] = {}
                    if not drc_layer in self.drc_dict[drc_type]:
                        self.drc_dict[drc_type][drc_layer] = []
                    else:
                        pass
                elif line.startswith('Bounds'):
                    drc_area = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
                    # print(drc_area)
                    self.drc_dict[drc_type][drc_layer].append([float(drc_area[0])*self.unit, float(drc_area[1])*self.unit, float(drc_area[2])*self.unit, float(drc_area[3])*self.unit])
                else:
                    pass
                
    """
    Function used by get_DRC. Compute DRC density in GCell grids.
    """
    
    def compute_DRC_density(self):
        drc_density_dict = {}
        overall_count = 0
        save_path = os.path.join(self.save_path, 'DRC')
        rpt_dir = os.path.join(save_path, 'rpt')
        for types in self.drc_dict:      # first layer of dict: key types
            type_count = 0
            drc_density_dict[types] = np.zeros(self.gcell_size)
            save_dir = os.path.join(save_path, 'DRC_seperated', types.replace(' ', ''), self.save_name)
            os.system("mkdir -p %s " % (os.path.dirname(save_dir)))
            for layer in self.drc_dict[types]:    # second: metal layer 
                for drc in self.drc_dict[types][layer]:      # drc_dict[types][layer] third: coordinate
                    drc_left, drc_lower, drc_right, drc_upper = drc
                    drc_x_left_gcell = bisect.bisect_left(self.gcell_coordinate_x, drc_left)
                    drc_y_lower_gcell = bisect.bisect_left(self.gcell_coordinate_y, drc_lower)
                    drc_x_right_gcell = bisect.bisect_left(self.gcell_coordinate_x, drc_right)
                    drc_y_upper_gcell = bisect.bisect_left(self.gcell_coordinate_y, drc_upper)
                    drc_gcell = [drc_x_left_gcell, drc_y_lower_gcell, drc_x_right_gcell, drc_y_upper_gcell]
                    drc_density_dict[types] = self.compute_density(drc_density_dict[types], drc_gcell)
                    type_count += 1
                    overall_count += 1
            with open(rpt_dir, 'a') as write:
                write.writelines('%s: %s\n' % (types, type_count))
            np.save(save_dir, drc_density_dict[types])  # save based on types

        with open(rpt_dir, 'a') as write:
            write.writelines('all: %s\n' % overall_count)

        overall_drc_density = np.zeros(self.gcell_size)
        for i in drc_density_dict:
            overall_drc_density += drc_density_dict[i]
        save_dir = os.path.join(save_path, 'DRC_all', self.save_name)
        os.system("mkdir -p %s " % (os.path.dirname(save_dir)))
        np.save(save_dir, overall_drc_density)  
        
    """
    Get IR drop features.
    """
    
    def get_IR_features(self):
        self.read_twf()
        self.read_power()
        self.get_power_map()
        self.get_IR()
                
    """
    Function used by get_IR_features. Read timing window file to get timing window.
    """
    
    def read_twf(self):
        with open(self.twf_path, 'r') as read_file: 
            for line in read_file:
                if "WAVEFORM" in line:
                    clk_data = line.split()
                    clk = int(float(clk_data[2]))
                    time_windows = np.linspace(-1, clk, self.n_time_window + 1)
                    time_windows = np.delete(time_windows, 0)
                elif "NET" in line:
                    if "CONSTANT" in line:
                        data = line.split()
                        name = data[2].replace('\\', '').replace('"', '')
                        pin_list = self.route_net_dict[name]
                        for cell_name in pin_list:
                            if cell_name not in self.tw_dict:
                                self.tw_dict[cell_name] = []
                            self.tw_dict[cell_name].append(0)
                    else:
                        data = line.split()

                        if data[2] == '*' or data[6] == '*':
                            name = data[1].replace('\\', '').replace('"', '')
                            pin_list = self.route_net_dict[name]
                            for cell_name in pin_list:
                                if cell_name not in self.tw_dict:
                                    self.tw_dict[cell_name] = []
                                self.tw_dict[cell_name].append(0)
                        else:
                            name = data[1].replace('\\', '').replace('"', '')
                            pin_list = self.route_net_dict[name]
                            time_arrive = data[2].split(':')
                            time_arrive.extend(data[6].split(':'))
                            time_arrive = [float(i) for i in time_arrive]
                            time_arrive_window = [min(time_arrive), max(time_arrive)]
                            tw_result = [bisect.bisect_left(time_windows, time_arrive_window[0]), bisect.bisect_left(time_windows, time_arrive_window[1])]
                            for cell_name in pin_list:
                                if cell_name not in self.tw_dict:
                                    self.tw_dict[cell_name] = []
                                self.tw_dict[cell_name].append(tw_result)
                
    """
    Function used by get_IR_features. Read instance power report.
    """
    
    def read_power(self):
        with open(self.power_path, 'r') as read_file:
            start = False
            read = False
            for line in read_file:
                if "Instance" in line:
                    start = True
                if start and line.startswith('Total'):
                    break
                if start:
                    if line.startswith('-'):
                        read = True
                    if read:
                        if len(line.split()) == 1:
                            name = line.split()[0].replace('\\', '')
                        elif len(line.split()) == 8:
                            if line.split()[-1] not in self.lef_dict or not self.lef_dict[line.split()[-1]]['type'] == 'std_cell':
                                continue
                            data = line.split()
                            if eval(data[0]) == 0:
                                if 'FILLER' in name:
                                    self.power_dict[name] = [0, float(data[2]), float(data[3]), float(data[4]),
                                                    'filler']
                                else:
                                    self.power_dict[name] = [0, float(data[2]), float(data[3]),float(data[4]), self.tw_dict[name]]

                            else:
                                self.power_dict[name] = [eval(data[1])/eval(data[0]), float(data[2]), float(data[3]), float(data[4]), self.tw_dict[name]]
                        elif len(line.split()) == 9:
                            if line.split()[-1] not in self.lef_dict or not self.lef_dict[line.split()[-1]]['type'] == 'std_cell':
                                continue
                            data = line.split()
                            name = data[0].replace('\\', '')
                            if eval(data[1]) == 0:
                                if 'FILLER' in name:
                                    self.power_dict[name] = [0, float(data[3]), float(data[4]), float(data[5]),
                                                    'filler']
                                else:
                                    self.power_dict[name] = [0, float(data[3]), float(data[4]), float(data[5]), self.tw_dict[name]]
                            else:
                                self.power_dict[name] = [eval(data[2])/eval(data[1]), float(data[3]), float(data[4]), float(data[5]), self.tw_dict[name]]
                
    """
    Function used by get_IR_features. Compute power map in GCell grids.
    """
    
    def get_power_map(self):
        window_shape = [self.n_time_window]
        window_shape.extend(self.gcell_size)
        self.power_t = np.zeros(window_shape)
        self.power_i = np.zeros(self.gcell_size)
        self.power_s = np.zeros(self.gcell_size)
        self.power_sca = np.zeros(self.gcell_size)
        self.power_all = np.zeros(self.gcell_size)
        power_map = np.zeros(self.gcell_size)
        for k, v in self.power_dict.items():
            if v[4] == 'filler':
                tw = [0]
            else:
                tw = v[4]
            instance = self.route_instance_dict[k]
            cell_size = self.lef_dict[instance[0]]['size']
            direction = instance_direction_rect(instance[2])
            cell_x_left = instance[1][0]
            cell_y_lower = instance[1][1]
            cell_x_right = cell_x_left + cell_size[0] * direction[0] + cell_size[1] * direction[1]
            cell_y_upper = cell_y_lower + cell_size[0] * direction[2] + cell_size[1] * direction[3]
            cell_x_left_gcell = bisect.bisect_left(self.gcell_coordinate_x, cell_x_left)
            cell_y_lower_gcell = bisect.bisect_left(self.gcell_coordinate_y, cell_y_lower)
            cell_x_right_gcell = bisect.bisect_left(self.gcell_coordinate_x, cell_x_right)
            cell_y_upper_gcell = bisect.bisect_left(self.gcell_coordinate_y, cell_y_upper)
            location = [cell_x_left, cell_y_lower, cell_x_right, cell_y_upper]
            location_gcell = [cell_x_left_gcell, cell_y_lower_gcell, cell_x_right_gcell, cell_y_upper_gcell]
            tmp_power_map = np.zeros(self.gcell_size)
            power_map += self.compute_density_with_overlap(tmp_power_map, location, location_gcell)
            n_pin = len(tw)
            self.power_i += power_map * v[1] * n_pin
            self.power_s += power_map * v[2] * n_pin
            self.power_sca += power_map * ((v[1] + v[2]) * v[0] + v[3]) * n_pin
            self.power_all += power_map * (v[1] + v[2] + v[3]) * n_pin
            if tw:
                for i in tw:
                    if i == 0:
                        pass
                    else:
                        self.power_t[i[0]:i[1]+1, :, :] += power_map * ((v[1] + v[2]) * v[0] + v[3])
        save(self.save_path, 'IR_drop/power_t', self.save_name, self.power_t)
        save(self.save_path, 'IR_drop/power_i', self.save_name, self.power_i)
        save(self.save_path, 'IR_drop/power_s', self.save_name, self.power_s)
        save(self.save_path, 'IR_drop/power_sca', self.save_name, self.power_sca)
        save(self.save_path, 'IR_drop/power_all', self.save_name, self.power_all)

    """
    Function used by get_IR_features. Read IR drop reports.
    """
    
    def get_IR(self):
        self.ir_map = np.zeros(self.gcell_size)
        with open(self.ir_path, 'r') as read_file:
            read = False
            first_flag = False
            for line in read_file:
                if line.startswith('Range'):
                    read = False
                if read:
                    data = line.split()
                    if not data[1] == 'M4':
                        first_flag = True
                    if first_flag:
                        gcell_x = bisect.bisect_left(self.gcell_coordinate_x, float(data[2]) * self.unit)
                        gcell_y = bisect.bisect_left(self.gcell_coordinate_y, float(data[3]) * self.unit)
                        if float(data[0]) > self.ir_map[gcell_x, gcell_y]:
                            self.ir_map[gcell_x, gcell_y] = float(data[0])   #v to mv *1000  % /100
                if line.startswith('ir'):
                    read = True
        save(self.save_path, 'IR_drop/IR_drop', self.save_name, self.ir_map)
        


    def get_pin_configuration_map(self):
        scale = 50.0
        pin_map_M1 = np.zeros((round(self.gcell_coordinate_x[-1]/scale), round(self.gcell_coordinate_y[-1]/scale)), dtype=np.int8)
        pin_map_M2 = np.zeros((round(self.gcell_coordinate_x[-1]/scale), round(self.gcell_coordinate_y[-1]/scale)), dtype=np.int8)
        for v in self.route_instance_dict.values():
            lef_dict = self.lef_dict_jnet[v[0]]
            instance_location_on_chip = [v[1][0], v[1][1]]
            direction = instance_direction_bottom_left(v[2])
            std_cell_x, std_cell_y = lef_dict['size']
            instance_location_on_chip[0] += direction[8] * std_cell_x + direction[9] * std_cell_y    #先把坐标转回原本的左下点
            instance_location_on_chip[1] += direction[10] * std_cell_x + direction[11] * std_cell_y

            for pin_name, pin_data in lef_dict['pin'].items():
                if pin_name == 'OBS':
                    continue
                elif pin_name == 'VDD' or pin_name == 'VSS':
                    continue
                for layer, rects in pin_data.items():
                    if not 'M' in layer:
                        continue
                    for pin in rects:
                        pin_left = instance_location_on_chip[0] + pin[0] * direction[4] + pin[1] * direction[5] + pin[2] * direction[0] + pin[3] * direction[1]
                        pin_lower = instance_location_on_chip[1] + pin[0] * direction[6] + pin[1] * direction[7] + pin[2] * direction[2] + pin[3] * direction[3]
                        pin_right = instance_location_on_chip[0] + pin[2] * direction[4] + pin[3] * direction[5] + pin[0] * direction[0] + pin[1] * direction[1]
                        pin_upper = instance_location_on_chip[1] + pin[2] * direction[6] + pin[3] * direction[7] + pin[0] * direction[2] + pin[1] * direction[3]
                        pin_left =  int(pin_left/scale-1)
                        pin_lower = int(pin_lower/scale-1)
                        pin_right = int(pin_right/scale-1)
                        pin_upper = int(pin_upper/scale-1)
                        if layer == 'M1':
                            pin_map_M1[pin_left:(pin_right+1), pin_lower:(pin_upper+1)] = 1
                        else:
                            pin_map_M2[pin_left:(pin_right+1), pin_lower:(pin_upper+1)] = 1
        save_path = os.path.join(self.save_path, 'pin_configure', self.save_name)
        os.system("mkdir -p %s " % (os.path.dirname(save_path)))
        np.savez_compressed(save_path, M1=pin_map_M1, M2=pin_map_M2)
