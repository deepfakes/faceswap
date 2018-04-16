import traceback
from pathlib import Path
from utils import Path_utils
import cv2
from tqdm import tqdm
from utils.AlignedPNG import AlignedPNG
from utils import image_utils
import shutil
import numpy as np
            
def main (input_dir, output_dir, aligned_dir, model_dir, model_name, **in_options):
    print ("Running converter.\r\n")
    
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        aligned_path = Path(aligned_dir)
        model_path = Path(model_dir)
        
        if not input_path.exists():
            print('Input directory not found. Please ensure it exists.')
            return

        if output_path.exists():
            for filename in Path_utils.get_image_paths(output_path):
                Path(filename).unlink()
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            
        if not aligned_path.exists():
            print('Aligned directory not found. Please ensure it exists.')
            return
            
        if not model_path.exists():
            print('Model directory not found. Please ensure it exists.')
            return

        import models 
        model = models.import_model(model_name)(model_path, **in_options)
        converter = model.get_converter(**in_options)
        
        input_path_image_paths = Path_utils.get_image_paths(input_path)
        aligned_path_image_paths = Path_utils.get_image_paths(aligned_path)
        
        alignments = {}
        for filename in tqdm(aligned_path_image_paths, desc= "Collecting alignments" ):
            a_png = AlignedPNG.load( str(filename) )
            if a_png is None:
                print ( "%s - no embedded data found." % (filename) )
                continue
            d = a_png.getFaceswapDictData()
            if d is None or d['source_filename'] is None or d['source_rect'] is None or d['source_landmarks'] is None:
                print ( "%s - no embedded data found." % (filename) )
                continue
            
            source_filename_stem = Path(d['source_filename']).stem
            if source_filename_stem not in alignments.keys():
                alignments[ source_filename_stem ] = []

            alignments[ source_filename_stem ].append ( np.array(d['source_landmarks']) )

        
        for filename in tqdm( input_path_image_paths, desc="Converting"):
            filename_path = Path(filename)
            output_filename_path = output_path / filename_path.name
         
            if filename_path.stem not in alignments.keys():                        
                if not model.is_debug():
                    print ( 'no faces found for %s, copying without faces' % (filename_path.name) )                
                    shutil.copy ( str(filename_path), str(output_filename_path) )                
            else:                    
                image = (cv2.imread(filename) / 255.0).astype('float32')
                faces = alignments[filename_path.stem]
                for image_landmarks in faces:                
                    image = converter.convert(image, image_landmarks, model.is_debug()) 
        
                    if model.is_debug():
                        for img in image:
                            cv2.imshow ('Debug convert', img )
                            cv2.waitKey(0)
                
                if not model.is_debug():
                    cv2.imwrite (str(output_filename_path), (image*255).astype(np.uint8) )
        
        model.finalize()
    except Exception as e:
        print ( 'Error: %s' % (str(e)))
        traceback.print_exc()
    
   
