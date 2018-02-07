import multiprocessing as mp

method = None

def pool_process(method_to_run, data, processes=None):
    global method
    if processes is None:
        processes = mp.cpu_count()
    method = method_to_run
    pool = mp.Pool(processes=processes)

    for i in pool.imap_unordered(runner, data):
        yield i if i is not None else 0
    
def runner(item):
    return method(item)
