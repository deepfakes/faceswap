try:
    from lib.faces_detect import detect_faces
except RuntimeError as e:
    raise Exception("possible out of memory!!!")

from plugins.PluginLoader import PluginLoader
import cv2
import os

def extract(input_path,  output_path):
    files = os.listdir(input_path)
    if not len(files): raise Exception("no files inside {0}!!!".format(input_path))
    if not os.path.exists(output_path): raise Exception("output directory {0} not exists!!!".format(input_path))
	
    extractor = PluginLoader.get_extractor("Align")()
    for n, _file in enumerate(files):
        print ("file {0}/{1}".format(n,len(files)))
        _file_id = os.path.join(input_path, _file)
        image = cv2.imread(_file_id)
        try:
            for (idx, face) in enumerate(detect_faces(image)):
                resized_image = extractor.extract(image, face, 256)
                output = output_path.format(_file)
                cv2.imwrite(output, resized_image)
        except Exception as e:
            print('Failed to extract from image: {}. Reason: {}'.format(_file, e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract faces')
    parser.add_argument('-i', '--input',
                        action="store",
                        dest='input directory path',
                        help='set the input directory path to source images',
                        default="")
    parser.add_argument('-o', '--output',
                        action="store",
                        dest='output directory path',
                        help='set the output directory path',
                        default="")
    _arg = parser.parse_args()
    extract(_arg.input, _arg.output)
    

