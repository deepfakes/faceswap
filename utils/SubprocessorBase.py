import traceback
from tqdm import tqdm
import multiprocessing
import time

class SubprocessorBase(object):
    def __init__(self, name): 
        self.name = name
        
    #overridable    
    def process_info_generator(self):
        #yield name, host_dict, client_dict - per process
        yield 'first process', {}, {}
        
    #overridable
    def get_no_process_started_message(self):
        return "No process started."

    #overridable
    def onHostGetProgressBarDesc(self):
        return "Processing"
        
    #overridable
    def onHostGetProgressBarLen(self):
        return 0
        
    #overridable        
    def onHostGetData(self):
        #return data here
        return None
    
    #overridable
    def onHostDataReturn (self, data):
        #input_data.insert(0, obj['data'])   
        pass
    
    #overridable
    def onClientInitialize(self, client_dict):
        #return fail message or None if ok
        return None
        
    #overridable
    def onClientFinalize(self):
        pass
        
    #overridable
    def onClientProcessData(self, data):
        #return result object
        return None  
        
    #overridable
    def onHostResult (self, data, result):
        #return count of progress bar update
        return 1
    
    #overridable
    def get_start_return(self):
        return None
    
    def start(self):
        #returns start_return
        
        self.processes = []
        
        for name, host_dict, client_dict in self.process_info_generator():            
            sq = multiprocessing.Queue()
            cq = multiprocessing.Queue()
            p = multiprocessing.Process(target=self.process, args=(sq,cq,client_dict))
            p.daemon = True
            p.start()
            self.processes.append ( { 'process' : p,
                                      'sq' : sq,
                                      'cq' : cq,
                                      'state' : 'busy',
                                      'sent_time': time.time(),
                                      'name': name,
                                      'host_dict' : host_dict
                                    } )

        while True:
            for p in self.processes[:]:
                while not p['cq'].empty():
                    obj = p['cq'].get()
                    obj_op = obj['op']
                        
                    if obj_op == 'init_ok':
                        p['state'] = 'free'
                    elif obj_op == 'error':                       
                        if obj['close'] == True:
                            p['process'].terminate()
                            p['process'].join()
                            self.processes.remove(p)
                            break                
                    
            if all ([ p['state'] == 'free' for p in self.processes ] ):
                break
                
        if len(self.processes) == 0:
            print ( self.get_no_process_started_message() )
            return
            
        progress_bar = tqdm( total=self.onHostGetProgressBarLen(), desc=self.onHostGetProgressBarDesc() )  
        
        try: 
            while True:
                for p in self.processes[:]:
                    while not p['cq'].empty():
                        obj = p['cq'].get()
                        obj_op = obj['op']
                        
                        if obj_op == 'success':             
                            data = obj['data']
                            result = obj['result']
                            
                            c = self.onHostResult (data, result)                        
                            if c > 0:
                                progress_bar.update(c)
                                
                            p['state'] = 'free'
                            
                        elif obj_op == 'error':
                            if 'data' in obj.keys():
                                self.onHostDataReturn ( obj['data'] )
            
                            if obj['close'] == True:
                                p['sq'].put ( {'op': 'close'} )
                                p['process'].join()
                                self.processes.remove(p)
                                break
                            p['state'] = 'free'
                            
                for p in self.processes[:]:
                    if p['state'] == 'free':
                        data = self.onHostGetData()
                        if data is not None:
                            p['sq'].put ( {'op': 'data', 'data' : data} )
                            p['sent_time'] = time.time()
                            p['sent_data'] = data
                            p['state'] = 'busy'                    
                        
                    elif p['state'] == 'busy':
                        if (time.time() - p['sent_time']) > 60:
                            print ( '%s doesnt response, terminating it.' % (p['name']) )
                            self.onHostDataReturn ( p['sent_data'] )                        
                            p['sq'].put ( {'op': 'close'} )
                            p['process'].join()
                            self.processes.remove(p)
                            
                if all ([p['state'] == 'free' for p in self.processes]):
                    break
                    
                time.sleep(0.005)
        except:
            print ("Exception occured in Subprocessor.start(): %s" % (traceback.format_exc()) )
        
        progress_bar.close()
        
        for p in self.processes[:]:
            p['sq'].put ( {'op': 'close'} )
            p['process'].join()
         
        return self.get_start_return()
         
    def process(self, sq, cq, client_dict):
        try:
            fail_message = self.onClientInitialize(client_dict)
        except:
            fail_message = 'Exception while initialization: %s' % (traceback.format_exc())
    
        if fail_message is None:
            cq.put ( {'op': 'init_ok'} )
        else:
            print (fail_message)
            cq.put ( {'op': 'error', 'close': True} )
            return 

        while True:
            obj = sq.get()
            obj_op = obj['op']

            if obj_op == 'data':
                data = obj['data']
                try:
                    result = self.onClientProcessData (data)                    
                    cq.put ( {'op': 'success', 'data' : data, 'result' : result} )                    
                except:
                    print ( 'Exception while process data: %s' % (traceback.format_exc()) )
                    cq.put ( {'op': 'error', 'close': True, 'data' : data } )
            elif obj_op == 'close':
                break
                
            time.sleep(0.005)
                
        self.onClientFinalize()