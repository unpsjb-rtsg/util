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


def plot_cvst(df, use_qt=False):
    if use_qt:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtGui, QtCore

        app = QtGui.QApplication([])
        win = pg.GraphicsWindow()
        win.resize(500,500)
        win.setWindowTitle('C vs T')

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        p = win.addPlot(title=file)
        p.plot(df["c"], df["t"], pen=None, symbol='o', symbolPen=None, symbolSize=10, symbolBrush=(100, 100, 255, 50))
        p.setLabel('left', "T")
        p.setLabel('bottom', "C")
        p.setLogMode(x=False, y=False)

        if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
            pg.QtGui.QApplication.exec_()
    else:
        df.plot.scatter(x='c', y='t')
        plt.show(block=True)


def get_args():
    """ Command line arguments """
    parser = ArgumentParser(description="Inspect sets of rts contained in hdfs store file(s) as pandas DataFrames.")
    parser.add_argument("files", help="XML file with RTS", nargs="+", type=str)
    parser.add_argument("--key", help="DataFrame key in store", default=None, type=str)
    parser.add_argument("--list", help="list keys in store", default=False, action="store_true")
    parser.add_argument("--get-rts-count", help="count number of rts in key", default=False, action="store_true")
    parser.add_argument("--print-rts", help="print rts in df specified by key", default=None, type=int)
    parser.add_argument("--get-info", help="retrive metadata for the DataFrame specified by key", default=False, action="store_true")
    parser.add_argument("--plot", help="Plot DataFrame", default=None, choices=["u"])
    return parser.parse_args()


def hdf_store(args, df_dict):
    with pd.HDFStore(args.save, complevel=9, complib='blosc') as store:
        for key, df in df_dict.items():
            if key in store:
                if args.reuse_key is False:
                    sys.stderr.write("{0} -- key already exists (use --reuse-key).\n".format(args.save_key))
                    exit(1)
                else:
                    store.remove(args.save_key)

            # save the results into the store
            store.put(key, df, format='table')


def main():
    args = get_args()

    if args.list:
        # list stored dataframes
        for file in args.files:
            if not os.path.isfile(file):
                print("{0}: file not found.".format(file))
                continue

            with pd.HDFStore(file) as store:
                for key in store.keys():
                    print(key)

        exit(0)

    if args.key is None:
        sys.stderr.write("No key specified.\n")
        exit(1)
        

    for file in args.files:
        if not os.path.isfile(file):
            sys.stderr.write("{0}: file not found\n.".format(file))
            continue

        with pd.HDFStore(file) as store:
            if args.key not in store.keys():
                sys.stderr.write("key {0} not found.\n".format(args.key))
                continue
                
            # retrive the specified DataFrame from the store
            df = store.select(args.key)
            
            # do the selected processing
            if df is not None:
                if args.get_rts_count:
                    print("{0}: {1} rts.".format(args.key, len(df.groupby(["rts_id"]))))
                
                if args.print_rts:
                    print("{0}:\n{1}".format(args.key, df[df["rts_id"] == args.print_rts]))
                
                if args.get_info:
                    metadata = store.get_storer(args.key).attrs.metadata
                    if metadata:
                        print("{0} metadata:".format(args.key))
                        for k, v in metadata.items():
                            print("{0}: {1}".format(k, v))
                    else:
                        print("{0}: No metadata available.".format(args.key))
                
                if args.plot == "u":
                    plot_cvst(df)
                    
    

if __name__ == '__main__':
    main()
