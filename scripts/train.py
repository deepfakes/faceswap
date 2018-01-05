import cv2
import numpy
import time

from lib.utils import get_image_paths
from lib.cli import FullPaths
from plugins.PluginLoader import PluginLoader

class TrainingProcessor(object):
    arguments = None

    def __init__(self, subparser, command, description='default'):
        self.parse_arguments(description, subparser, command)

    def process_arguments(self, arguments):
        self.arguments = arguments
        print("Model A Directory: {}".format(self.arguments.input_A))
        print("Model B Directory: {}".format(self.arguments.input_B))
        print("Training data directory: {}".format(self.arguments.model_dir))

        self.process()

    def parse_arguments(self, description, subparser, command):
        parser = subparser.add_parser(
            command,
            help="This command trains the model for the two faces A and B.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )

        parser.add_argument('-A', '--input-A',
                            action=FullPaths,
                            dest="input_A",
                            default="input_A",
                            help="Input directory. A directory containing training images for face A.\
                             Defaults to 'input'")
        parser.add_argument('-B', '--input-B',
                            action=FullPaths,
                            dest="input_B",
                            default="input_B",
                            help="Input directory. A directory containing training images for face B.\
                             Defaults to 'input'")
        parser.add_argument('-m', '--model-dir',
                            action=FullPaths,
                            dest="model_dir",
                            default="models",
                            help="Model directory. This is where the training data will \
                                be stored. Defaults to 'model'")
        parser.add_argument('-p', '--preview',
                            action="store_true",
                            dest="preview",
                            default=False,
                            help="Show preview output. If not specified, write progress \
                            to file.")
        parser.add_argument('-v', '--verbose',
                            action="store_true",
                            dest="verbose",
                            default=False,
                            help="Show verbose output")
        parser.add_argument('-s', '--save-interval',
                            type=int,
                            dest="save_interval",
                            default=100,
                            help="Sets the number of iterations before saving the model.")
        parser.add_argument('-w', '--write-image',
                            action="store_true",
                            dest="write_image",
                            default=False,
                            help="Writes the training result to a file even on preview mode.")
        parser = self.add_optional_arguments(parser)
        parser.set_defaults(func=self.process_arguments)

    def add_optional_arguments(self, parser):
        # Override this for custom arguments
        return parser

    def process(self):
        variant = "GAN"
        
        print('Loading data, this may take a while...')
        model = PluginLoader.get_model(variant)(self.arguments.model_dir)
        model.load(swapped=False)

        images_A = get_image_paths(self.arguments.input_A)
        images_B = get_image_paths(self.arguments.input_B)
        trainer = PluginLoader.get_trainer(variant)(model, images_A, images_B)

        try:
            print('Starting. Press "q" to stop training and save model')

            for epoch in range(1, 1000000): # Note starting at 1 may change behavior of tests on "epoch % n == 0"
                if self.arguments.verbose:
                    print("Iteration number {}".format(epoch))
                    start_time = time.time()

                sample_gen = trainer.train_one_step(epoch)

                if epoch % self.arguments.save_interval == 0:
                    model.save_weights()
                    self.show(sample_gen)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    model.save_weights()
                    exit()
                if self.arguments.verbose:
                    end_time = time.time()
                    time_elapsed = int(round((end_time - start_time)))
                    m, s = divmod(time_elapsed, 60)
                    h, m = divmod(m, 60)
                    print("Iteration done in {:02d}h{:02d}m{:02d}s".format(h, m, s))
        except KeyboardInterrupt:
            try:
                model.save_weights()
            except KeyboardInterrupt:
                print('Saving model weights has been cancelled!')
            sys.exit(0)

    def show(self, image_gen):
        if self.arguments.preview:
            cv2.imshow('', image_gen())
        elif self.arguments.write_image:
            cv2.imwrite('_sample.jpg', image_gen())
