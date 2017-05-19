import os
import math
import sys
import pandas as pd
import numpy as np
from tabulate import tabulate
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def get_rts_from_xml(xml_file, start, count):
    import xml.etree.cElementTree as et

    context = et.iterparse(xml_file, events=('start', 'end'))
    context = iter(context)
    event, root = context.__next__()

    rts_count = 0
    rts_limit = start + count

    rts_id, tasks = 0, []
    rts_list = []

    xml_uf = int(float(root.get("u")))
    xml_rts_size = int(float(root.get("n")))
    xml_set_size = int(float(root.get("size")))

    for event, elem in context:
        if elem.tag == 'S':
            if event == 'start':
                rts_id = int(float(elem.get("count")))
                tasks = []
            
            if event == 'end':
                if rts_count >= rts_limit:
                    break
                
                rts_count += 1

                if rts_count >= start:                
                    # schedulability analysis
                    sched, wcrt = rta3(tasks)

                    # add wcrt
                    for t, r in zip(tasks, wcrt):
                        t["wcrt"] = r

                    rts = {'id': rts_id, 'sched': sched, 'tasks': tasks}

                    rts_list.append(rts)

        if event == 'start' and elem.tag == 'i':
            task = elem.attrib
            for k, v in task.items():
                task[k] = int(float(v))
            task["rts_id"] = rts_id
            tasks.append(task)        

        root.clear()

    del context

    return rts_count, xml_set_size, xml_rts_size, xml_uf, rts_list


def rta3(rts):
    """
    RTA3 -- "Computational Cost Reduction for Real-Time Schedulability Tests Algorithms"
    http://ieeexplore.ieee.org/document/7404899/
    """
    wcrt = [0] * len(rts)
    a = [0] * len(rts)
    i = [0] * len(rts)
    schedulable = True
    flag = True

    for idx, task in enumerate(rts):
        a[idx] = task["C"]
        i[idx] = task["T"]

    t = rts[0]["C"]
    wcrt[0] = rts[0]["C"]

    for idx, task in enumerate(rts[1:], 1):
        t_mas = t + task["C"]

        while schedulable:
            t = t_mas

            for jdx, jtask in zip(range(len(rts[:idx]) - 1, -1, -1), reversed(rts[:idx])):
                if t_mas > i[jdx]:
                    tmp = math.ceil(t_mas / jtask["T"])
                    a_tmp = tmp * jtask["C"]

                    t_mas += (a_tmp - a[jdx])
                    
                    if t_mas > task["D"]:
                        schedulable = False
                        break

                    a[jdx] = a_tmp
                    i[jdx] = tmp * jtask["T"]

            if t == t_mas:
                break

        wcrt[idx] = t

        if not schedulable:
            wcrt[idx] = 0
            break

    return [schedulable, wcrt]


def get_args():
    """ Command line arguments """
    parser = ArgumentParser(description="Save rts in xml file(s) into a hdfs store as pandas DataFrames.")
    parser.add_argument("files", help="XML file with rts", nargs="+", type=str)
    parser.add_argument("--start", help="rts where start", type=int, default=0)
    parser.add_argument("--count", help="number of rts to save in the store", type=int, default=1)
    parser.add_argument("--save", help="Save the results into the specified HDF5 store",
                        default=None, type=str)
    parser.add_argument("--save-key", help="Key for storing the results into a HDF5 store",
                        default=None, type=str)
    parser.add_argument("--reuse-key", help="Replace DataFrame under key in the store",
                        default=False, action="store_true")
    parser.add_argument("--dist-type", help="Distribution", type=str, choices=["r1", "r2", "r3", "r4"], default=None)
    parser.add_argument("--range", help="Range", type=str, choices=["p25_1000", "p25_10000", "p25_100000", "p25_1000000"], default=None)
    parser.add_argument("--verbose", help="Show extra information.", default=False, action="store_true")
    return parser.parse_args()


def hdf_store(args, key, df, metadata):
    with pd.HDFStore(args.save, complevel=9, complib='blosc') as store:        
        if key in store:
            if args.reuse_key is False:
                sys.stderr.write("{0} -- key already exists (use --reuse-key).\n".format(args.save_key))
                exit(1)
            else:
                store.remove(args.save_key)

        # save the results into the store
        store.put(key, df, format='table')

        # add metadata
        store.get_storer(key).attrs.metadata = metadata


def main():
    args = get_args()

    if args.dist_type is None:
        sys.stderr.write("Error: no distribution type specified.\n")
        exit(1)

    if args.range is None:
        sys.stderr.write("Error: no range specified.\n")
        exit(1)

    # key hierarchy: dist_type/range/task_cnt/uf -- example: r1/p25-1000/n10/u70

    for file in args.files:
        if not os.path.isfile(file):
            print("{0}: file not found.".format(file))
            sys.exit(1)

        if args.verbose:
            sys.stderr.write("Exporting file: {0}\n".format(file))

        # evaluate the methods with the rts in file        
        rts_count, xml_set_size, xml_rts_size, xml_uf, rts_list = get_rts_from_xml(file, args.start, args.count)

        # additional metadata to attach to the dataframe in the store
        metadata = {'xml': file, 'uf': xml_uf, 'dist': args.dist_type, 'range': args.range, 'size': rts_count}

        tmp_list = []
        for rts in rts_list:
            for task in rts["tasks"]:
                tmp_list.append(task)

        # lower case all column names
        df = pd.DataFrame.from_dict(tmp_list)
        df.columns = map(str.lower, df.columns)
        df.rename(columns={'nro': 'task_id'}, inplace=True)
        
        # key
        key = "/{0}/{1}/n{2}/u{3}".format(args.dist_type, args.range, xml_rts_size, xml_uf)
        
        # save dataframes into hdfs store
        if args.save:
            hdf_store(args, key, df, metadata)
    

if __name__ == '__main__':
    main()
