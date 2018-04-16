import sys

def set_process_lowest_prio():
    if sys.platform[0:3] == 'win':
        from ctypes import windll
        from ctypes import wintypes
        
        GetCurrentProcess = windll.kernel32.GetCurrentProcess
        GetCurrentProcess.restype = wintypes.HANDLE
        
        SetPriorityClass = windll.kernel32.SetPriorityClass
        SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        SetPriorityClass ( GetCurrentProcess(), 0x00000040 )
        