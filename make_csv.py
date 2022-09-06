import os
import csv


def make_csv():

    path = './rpt'

    count = 1

    for file_path in os.listdir(path):
        name = os.path.basename(file_path)
        if not len(name.split('-')) == 4:
            continue
        setup1, setup2, setup3, setup4 = name.split('-')

        part1 = setup1.split('_')
        part2 = setup2.split('_')
        part4 = setup4.split('_')

        design = None
        if part2[2] == 'f1':
            design = 'RISCY-FPU'
        elif part2[3] == 'z1':
            design = 'zero-riscy'
        elif part2[2] == 'f0' and part2[3] == 'z0':
            design = 'RISCY'
        else:
            raise ValueError

        modification = None
        if part1[1] == '7t' or part1[1] == 'xo':
            modification = 'a'
        elif part1[1] == 'macro':
            modification = 'b'

        filler = None
        if len(part1) == 2:
            filler = 0
        elif len(part1) == 3:
            filler = 1
        else:
            raise ValueError

        sram = None
        if part2[4] == 's1':
            sram = 1
        elif part2[4] == 's2':
            sram = 2
        elif part2[4] == 's3':
            sram = 3
        else:
            raise ValueError
        
        clock = part2[1]

        FPutil = part4[5].replace('FP', '')

        PGsetting = setup3[0]

        macro_placement = None
        if len(part1) == 2 and (part1[1] == '7t' or part1[1] == 'xo'):
            if part4[6] == 'high':
                if part4[8] == '0':
                    macro_placement = 1
                elif part4[8] == '10':
                    macro_placement = 2
                elif part4[8] == '50':
                    macro_placement = 4
            elif  part4[6] == 'medium':
                    macro_placement = 3
        else:
            if part4[6] == 'high':
                if part4[8] == '10':
                    macro_placement = 1
                elif part4[8] == '50':
                    macro_placement = 2
            elif  part4[6] == 'medium':
                    macro_placement = 3


        name_list = [design, modification, sram, clock, FPutil, macro_placement, PGsetting, filler]
        new_name = "{0[0]}-{0[1]}-{0[2]}-c{0[3]}-u{0[4]}-m{0[5]}-p{0[6]}-f{0[7]}".format(name_list)

        row = [count, new_name]
        row.extend(name_list)
        with open('name_reference.csv', 'a') as f:
            f_csv = csv.writer(f, delimiter=',')
            f_csv.writerow(row)
        count += 1


make_csv()
