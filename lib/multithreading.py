import multiprocessing as mp

method = None

def pool_process(method_to_run, data):
    global method
    method = method_to_run
    pool = mp.Pool()
    for i in pool.imap_unordered(runner, data):
        yield(i)
    
def runner(item):
    return method(item)