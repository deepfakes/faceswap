import os
import sys
import argparse
from utils import Path_utils
from utils import os_utils
from pathlib import Path

if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 2):
    raise Exception("This program requires at least Python 3.2")

class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))
            
if __name__ == "__main__":
    os_utils.set_process_lowest_prio()

    parser = argparse.ArgumentParser()    
    parser.add_argument('--tf-suppress-std', action="store_true", dest="tf_suppress_std", default=False, help="Suppress tensorflow initialization info. May not works on some python builds such as anaconda python 3.6.4. If you can fix it, you are welcome.")
 
    subparsers = parser.add_subparsers()
    
    def process_extract(arguments):
        from mainscripts import Extractor        
        Extractor.main (
            input_dir=arguments.input_dir, 
            output_dir=arguments.output_dir, 
            debug=arguments.debug,
            detector=arguments.detector,
            multi_gpu=arguments.multi_gpu,
            manual_fix=arguments.manual_fix)
        
    extract_parser = subparsers.add_parser( "extract", help="Extract the faces from a pictures.")
    extract_parser.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    extract_parser.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted files will be stored.")
    extract_parser.add_argument('--debug', action="store_true", dest="debug", default=False, help="Writes debug images to [output_dir]_debug\ directory.")    
    extract_parser.add_argument('--detector', dest="detector", choices=['dlib','mt','manual'], default='dlib', help="Type of detector. Default 'dlib'. 'mt' (MTCNNv1) - faster, better, almost no jitter, perfect for gathering thousands faces for src-set. It is also good for dst-set, but can generate false faces in frames where main face not recognized! In this case for dst-set use either 'dlib' with '--manual-mode misses' or 'manual' detector. Manual detector suitable only for dst-set.")
    extract_parser.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=False, help="Enables multi GPU.")
    extract_parser.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False, help="Enables manual extract only frames where faces were not recognized.")
    extract_parser.set_defaults (func=process_extract)
    
    def process_sort(arguments):        
        from mainscripts import Sorter
        Sorter.main (input_path=arguments.input_dir, sort_by_method=arguments.sort_by_method)
        
    sort_parser = subparsers.add_parser( "sort", help="Sort faces in a directory.")     
    sort_parser.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    sort_parser.add_argument('--by', required=True, dest="sort_by_method", help="Sort by method.", choices=("blur", "face", "face-dissim", "face-yaw", "hist", "hist-dissim"))
    sort_parser.set_defaults (func=process_sort)
    
    def process_train(arguments):      
    
        if 'ODFS_TARGET_EPOCH' in os.environ.keys():
            arguments.target_epoch = int ( os.environ['ODFS_TARGET_EPOCH'] )
    
        if 'ODFS_BATCH_SIZE' in os.environ.keys():
            arguments.batch_size = int ( os.environ['ODFS_BATCH_SIZE'] )
    
    
        from mainscripts import Trainer
        Trainer.main (
            training_data_src_dir=arguments.training_data_src_dir, 
            training_data_dst_dir=arguments.training_data_dst_dir, 
            model_path=arguments.model_dir, 
            model_name=arguments.model_name,
            debug              = arguments.debug,
            #**options
            batch_size         = arguments.batch_size,
            write_preview_history=arguments.write_preview_history,
            target_epoch       = arguments.target_epoch,
            save_interval_min  = arguments.save_interval_min,
            force_best_gpu_idx = arguments.force_best_gpu_idx,
            multi_gpu          = arguments.multi_gpu
            )
        
    train_parser = subparsers.add_parser( "train", help="Trainer") 
    train_parser.add_argument('--training-data-src-dir', required=True, action=fixPathAction, dest="training_data_src_dir", help="Dir of src-set.")
    train_parser.add_argument('--training-data-dst-dir', required=True, action=fixPathAction, dest="training_data_dst_dir", help="Dir of dst-set.")
    train_parser.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
    train_parser.add_argument('--model', required=True, dest="model_name", choices=Path_utils.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Type of model")
    train_parser.add_argument('--write-preview-history', action="store_true", dest="write_preview_history", default=False, help="Enable write preview history.")
    train_parser.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug training.")    
    train_parser.add_argument('--batch-size', type=int, dest="batch_size", default=0, help="Model batch size. Default - auto. Environment variable: ODFS_BATCH_SIZE.") 
    train_parser.add_argument('--target-epoch', type=int, dest="target_epoch", default=0, help="Train until target epoch. Default - unlimited. Environment variable: ODFS_TARGET_EPOCH.")
    train_parser.add_argument('--save-interval-min', type=int, dest="save_interval_min", default=10, help="Save interval in minutes. Default 10.") 
    train_parser.add_argument('--force-best-gpu-idx', type=int, dest="force_best_gpu_idx", default=-1, help="Force to choose this GPU idx as best.")
    train_parser.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=False, help="MultiGPU option. It will select only same GPU models.")
    train_parser.set_defaults (func=process_train)
    
    def process_convert(arguments):        
        from mainscripts import Converter
        Converter.main (
            input_dir=arguments.input_dir, 
            output_dir=arguments.output_dir, 
            aligned_dir=arguments.aligned_dir,
            model_dir=arguments.model_dir, 
            model_name=arguments.model_name, 
            debug = arguments.debug,
            mode = arguments.mode,
            mask_type = arguments.mask_type
            )
        
    convert_parser = subparsers.add_parser( "convert", help="Converter") 
    convert_parser.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    convert_parser.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the converted files will be stored.")
    convert_parser.add_argument('--aligned-dir', required=True, action=fixPathAction, dest="aligned_dir", help="Aligned directory. This is where the aligned files stored.")
    convert_parser.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
    convert_parser.add_argument('--model', required=True, dest="model_name", choices=Path_utils.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Type of model")
    convert_parser.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug converter.")        
    convert_parser.add_argument('--mode',  dest="mode", choices=['seamless','hist-match'], default='seamless', help="Face overlaying mode. Seriously affects result.")
    convert_parser.add_argument('--mask-type',  dest="mask_type", choices=['predicted','dst','intersect'], default='predicted', help="Seriously affects result.")
    
    convert_parser.set_defaults(func=process_convert)

    def bad_args(arguments):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)
    
    arguments = parser.parse_args()
    if arguments.tf_suppress_std:
        os.environ['TF_SUPPRESS_STD'] = '1'
    arguments.func(arguments)


'''
import code
code.interact(local=dict(globals(), **locals()))
'''