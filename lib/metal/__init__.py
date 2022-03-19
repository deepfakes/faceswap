from typing import List
import os
import psutil # used for getting GPU memory
import tensorflow as tf

class Constants:
    class System:
        ARCH = 'arm64'
        DEVICE_TYPE = 'GPU'
        SET_MEMORY_GROWTH = True
    class CUDA:
        DRIVER_VERSION_UNSUPPORTED = 0

def _dbg_check_mem():
    print("==========================================")
    print(tf.config.experimental.get_memory_info('GPU:0'))
    print(tf.config.list_logical_devices())
    print("==========================================")

def _validate_metal():
    # Validate a GPU exists
    assert(len(tf.config.experimental.list_physical_devices('GPU')) > 0)

    # Validate Metal device is working
    with tf.device('GPU:0'):
        assert(tf.math.add(1.0, 2.0) == 3.0)

def init(device_type: str = 'GPU') -> None:
    #_validate_metal()

    os.environ['DISPLAY'] = ':0'
    try:
        os.system('open -a XQuartz')
    except Exception:
        pass
    Constants.System.DEVICE_TYPE = device_type

    #for device in get_devices():
    #    tf.config.experimental.set_memory_growth(device, Constants.System.SET_MEMORY_GROWTH)
    
    _dbg_check_mem()

def get_devices() -> List[tf.config.PhysicalDevice]:
    return tf.config.list_physical_devices(device_type=Constants.System.DEVICE_TYPE)

def get_device_count() -> int:
    return len(get_devices())

def get_handles() -> list:
    return list(range(get_device_count()))

def get_driver_version() -> int:
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html
    return Constants.CUDA.DRIVER_VERSION_UNSUPPORTED

def get_device_names() -> List[str]:
    return [d.name for d in get_devices()]

def get_memory_info(handle: int) -> int:
    # Does not work:
    #   tf.config.experimental.get_memory_info('GPU:0')
    # So, using psutil instead.
    # We can just grab the total memory, as it's shared between
    # the CPU and the GPU. There is no dedicated VRAM.
    return psutil.virtual_memory().total / get_device_count()