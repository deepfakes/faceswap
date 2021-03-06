��    P      �  k         �  u  �    ?  �   V	  �   
  �   �
     T  <   Y  �   �  �   !  9   �  `    �   n  f   3     �  w   �  �   "        .       @  �   O  �  N  �   �  �   �  �   )  D  �  �   	  [   �  ?  �  �   ,  �   �  C  �  C    c   J  8  �  J   �  )  2   B   \!  .   �!  F   �!  J   "     `"  �  h"  �   (  =  �(  �   +  �  �+  �  �0  �  �2  c  [6    �:     �<  �   �<  s   d=  6   �=  3   >  C   C>  F   �>  N   �>  B   ?  �   `?  �  �?  �  �A  �   <C  �   �C  �   �D  L   WE    �E  �   �F  m   ZG  c   �G  }   ,H     �H     �H     �H     �H     �H     �H  	   �H     �H  s  �H  �  bJ  2  M  {  IO  :  �P  :   R     ;S  v   HS  �   �S  (  �T  d   �U  �  CV  +  �X  �   Z     �Z  �   �Z  �  �[     C]  +  c]     �_  �  �_  �  �a  a  qd  �   �e  �   �f  ]  �g  }  )j  �   �k  R  fl  >  �n  �  �o  �  �q     �s  y   �u  ]  v  �   zx  |  y  S   �{  @   �{  �   |  �   �|     }}  �	  �}  �  �  b  ��  :  �  �  N�  �  �    �  z  �  �  ��     ��    ��  �   ��  j   c�  d   ί  �   3�  �   ��  �   ?�  �   ѱ  �   T�  �  6�  r  ζ  *  A�  �  l�  r  8�  �   ��  <  o�    ��  �   ��  �   ��    �     �     6�     ?�  
   L�  /   W�     ��     ��     ��            L      0   )   N   "          -      1             D                 7   &       4       G   E   P          6   C   H   !            $   (                      ;           K   %   B                  #       8                           9            ,         :       M   @   5   *   
   '      <   	   2   A   J      .       3   F       /   >   =                 O   I   +       ?           Automatically save the alignments file after a set amount of frames. By default the alignments file is only saved at the end of the extraction process. NB: If extracting in 2 passes then the alignments file will only start to be saved out during the second pass. WARNING: Don't interrupt the script when writing the file because it might get corrupted. Set to 0 to turn off Batch size. This is the number of images processed through the model for each side per iteration. NB: As the model is fed 2 sides at a time, the actual number of images within the model at any one time is double the number that you set here. Larger batches require more GPU RAM. Color augmentation helps make the model less susceptible to color differences between the A and B sets, at an increased training time cost. Enable this option to disable color augmentation. DEPRECATED - This option will be removed in a future update. Path to alignments file for training set A. Defaults to <input-A>/alignments.json if not provided. DEPRECATED - This option will be removed in a future update. Path to alignments file for training set B. Defaults to <input-B>/alignments.json if not provided. Data Disable multiprocessing. Slower but less resource intensive. Disables TensorBoard logging. NB: Disabling logs means that you will not be able to use the graph or analysis for this session in the GUI. Don't run extraction in parallel. Will run each part of the extraction process separately (one after the other) rather than all at the smae time. Useful if VRAM is at a premium. Draw landmarks on the ouput faces for debugging purposes. Enable On-The-Fly Conversion. NOT recommended. You should generate a clean alignments file for your destination video. However, if you wish you can generate the alignments on-the-fly by enabling this option. This will use an inferior extraction pipeline and will lead to substandard results. If an alignments file is found, this option will be ignored. Extract every 'nth' frame. This option will skip frames when extracting faces. For example a value of 1 will extract faces from every frame, a value of 10 will extract faces from every 10th frame. Extract faces from image or video sources.
Extraction plugins can be configured in the 'Settings' Menu Face Processing Filters out faces detected below this size. Length, in pixels across the diagonal of the bounding box. Set to 0 for off For use with the optional nfilter/filter files. Threshold for positive face recognition. Lower values are stricter. NB: Using face filter will significantly decrease extraction speed and its accuracy cannot be guaranteed. Frame Processing Frame ranges to apply transfer to e.g. For frames 10 to 50 and 90 to 100 use --frame-ranges 10-50 90-100. Frames falling outside of the selected range will be discarded unless '-k' (--keep-unchanged) is selected. NB: If you are converting from images, then the filenames must end with the frame-number! Global Options If a face isn't found, rotate the images to try to find a face. Can find more faces at the cost of extraction speed. Pass in a single number to use increments of that size up to 360, or pass in a list of numbers to enumerate exactly what angles to check. If you have not cleansed your alignments file, then you can filter out faces by defining a folder here that contains the faces extracted from your input files/video. If this folder is defined, then only faces that exist within your alignments file and also exist within the specified folder will be converted. Leaving this blank will convert all faces that exist within the alignments file. Input directory or video. Either a directory containing the image files you wish to process or path to a video file. NB: This should be the source video/frames NOT the source faces. Input directory. A directory containing training images for face A. This is the original face, i.e. the face that you want to remove and replace with face B. Input directory. A directory containing training images for face B. This is the swap face, i.e. the face that you want to place onto the head of person A. Length of training in iterations. This is only really used for automation. There is no 'correct' number of iterations a model should be trained for. You should stop training when you are happy with the previews. However, if you want the model to stop automatically at a set number of iterations, you can set that value here. Log level. Stick with INFO or VERBOSE unless you need to file an error report. Be careful with TRACE as it will generate a lot of data Model directory. The directory containing the trained model you wish to use for conversion. Model directory. This is where the training data will be stored. You should always specify a new folder for new models. If starting a new model, select either an empty folder, or a folder which does not exist (which will be created). If continuing to train an existing model, specify the location of the existing model. Only required if converting from images to video. Provide The original video that the source frames were extracted from (for extracting the fps and audio). Optional for creating a timelapse. Timelapse will save an image of your selected faces into the timelapse-output folder at every save iteration. If the input folders are supplied but no output folder, it will default to your model folder /timelapse/ Optional for creating a timelapse. Timelapse will save an image of your selected faces into the timelapse-output folder at every save iteration. This should be the input folder of 'A' faces that you would like to use for creating the timelapse. You must also supply a --timelapse-output and a --timelapse-input-B parameter. Optional for creating a timelapse. Timelapse will save an image of your selected faces into the timelapse-output folder at every save iteration. This should be the input folder of 'B' faces that you would like to use for creating the timelapse. You must also supply a --timelapse-output and a --timelapse-input-A parameter. Optional path to an alignments file. Leave blank if the alignments file is at the default location. Optionally filter out people who you do not wish to process by passing in an image of that person. Should be a front portrait with a single person in the image. Multiple images can be added space separated. NB: Using face filter will significantly decrease extraction speed and its accuracy cannot be guaranteed. Optionally overide the saved config with the path to a custom config file. Optionally select people you wish to process by passing in an image of that person. Should be a front portrait with a single person in the image. Multiple images can be added space separated. NB: Using face filter will significantly decrease extraction speed and its accuracy cannot be guaranteed. Output directory. This is where the converted files will be saved. Output to Shell console instead of GUI console Path to store the logfile. Leave blank to store in the faceswap folder Percentage amount to scale the preview by. 100%% is the model output size. Plugins R|Additional Masker(s) to use. The masks generated here will all take up GPU RAM. You can select none, one or multiple masks, but the extraction may take longer the more you select. NB: The Extended and Components (landmark based) masks are automatically generated on extraction.
L|vgg-clear: Mask designed to provide smart segmentation of mostly frontal faces clear of obstructions. Profile faces and obstructions may result in sub-par performance.
L|vgg-obstructed: Mask designed to provide smart segmentation of mostly frontal faces. The mask model has been specifically trained to recognize some facial obstructions (hands and eyeglasses). Profile faces may result in sub-par performance.
L|unet-dfl: Mask designed to provide smart segmentation of mostly frontal faces. The mask model has been trained by community members and will need testing for further description. Profile faces may result in sub-par performance.
The auto generated masks are as follows:
L|components: Mask designed to provide facial segmentation based on the positioning of landmark locations. A convex hull is constructed around the exterior of the landmarks to create a mask.
L|extended: Mask designed to provide facial segmentation based on the positioning of landmark locations. A convex hull is constructed around the exterior of the landmarks and the mask is extended upwards onto the forehead.
(eg: `-M unet-dfl vgg-clear`, `--masker vgg-obstructed`) R|Aligner to use.
L|cv2-dnn: A CPU only landmark detector. Faster, less resource intensive, but less accurate. Only use this if not using a GPU and time is important.
L|fan: Best aligner. Fast on GPU, slow on CPU. R|Detector to use. Some of these have configurable settings in '/config/extract.ini' or 'Settings > Configure Extract 'Plugins':
L|cv2-dnn: A CPU only extractor which is the least reliable and least resource intensive. Use this if not using a GPU and time is important.
L|mtcnn: Good detector. Fast on CPU, faster on GPU. Uses fewer resources than other GPU detectors but can often return more false positives.
L|s3fd: Best detector. Slow on CPU, faster on GPU. Can detect more faces and fewer false positives than other GPU detectors, but is a lot more resource intensive. R|Exclude GPUs from use by Faceswap. Select the number(s) which correspond to any GPU(s) that you do not wish to be made available to Faceswap. Selecting all GPUs here will force Faceswap into CPU mode.
L|{} R|Masker to use. NB: The mask you require must exist within the alignments file. You can add additional masks with the Mask Tool.
L|none: Don't use a mask.
L|components: Mask designed to provide facial segmentation based on the positioning of landmark locations. A convex hull is constructed around the exterior of the landmarks to create a mask.
L|extended: Mask designed to provide facial segmentation based on the positioning of landmark locations. A convex hull is constructed around the exterior of the landmarks and the mask is extended upwards onto the forehead.
L|vgg-clear: Mask designed to provide smart segmentation of mostly frontal faces clear of obstructions. Profile faces and obstructions may result in sub-par performance.
L|vgg-obstructed: Mask designed to provide smart segmentation of mostly frontal faces. The mask model has been specifically trained to recognize some facial obstructions (hands and eyeglasses). Profile faces may result in sub-par performance.
L|unet-dfl: Mask designed to provide smart segmentation of mostly frontal faces. The mask model has been trained by community members and will need testing for further description. Profile faces may result in sub-par performance. R|Performing normalization can help the aligner better align faces with difficult lighting conditions at an extraction speed cost. Different methods will yield different results on different sets. NB: This does not impact the output face, just the input to the aligner.
L|none: Don't perform normalization on the face.
L|clahe: Perform Contrast Limited Adaptive Histogram Equalization on the face.
L|hist: Equalize the histograms on the RGB channels.
L|mean: Normalize the face colors to the mean. R|Performs color adjustment to the swapped face. Some of these options have configurable settings in '/config/convert.ini' or 'Settings > Configure Convert Plugins':
L|avg-color: Adjust the mean of each color channel in the swapped reconstruction to equal the mean of the masked area in the original image.
L|color-transfer: Transfers the color distribution from the source to the target image using the mean and standard deviations of the L*a*b* color space.
L|manual-balance: Manually adjust the balance of the image in a variety of color spaces. Best used with the Preview tool to set correct values.
L|match-hist: Adjust the histogram of each color channel in the swapped reconstruction to equal the histogram of the masked area in the original image.
L|seamless-clone: Use cv2's seamless clone function to remove extreme gradients at the mask seam by smoothing colors. Generally does not give very satisfactory results.
L|none: Don't perform color adjustment. R|Select which trainer to use. Trainers can be configured from the Settings menu or the config folder.
L|original: The original model created by /u/deepfakes.
L|dfaker: 64px in/128px out model from dfaker. Enable 'warp-to-landmarks' for full dfaker method.
L|dfl-h128: 128px in/out model from deepfacelab
L|dfl-sae: Adaptable model from deepfacelab
L|dlight: A lightweight, high resolution DFaker variant.
L|iae: A model that uses intermediate layers to try to get better details
L|lightweight: A lightweight model for low-end cards. Don't expect great results. Can train as low as 1.6GB with batch size 8.
L|realface: A high detail, dual density model based on DFaker, with customizable in/out resolution. The autoencoders are unbalanced so B>A swaps won't work so well. By andenixa et al. Very configurable.
L|unbalanced: 128px in/out model from andenixa. The autoencoders are unbalanced so B>A swaps won't work so well. Very configurable.
L|villain: 128px in/out model from villainguy. Very resource hungry (You will require a GPU with a fair amount of VRAM). Good for details, but more susceptible to color differences. R|The plugin to use to output the converted images. The writers are configurable in '/config/convert.ini' or 'Settings > Configure Convert Plugins:'
L|ffmpeg: [video] Writes out the convert straight to video. When the input is a series of images then the '-ref' (--reference-video) parameter must be set.
L|gif: [animated image] Create an animated gif.
L|opencv: [images] The fastest image writer, but less options and formats than other plugins.
L|pillow: [images] Slower than opencv, but has more options and supports more formats. Saving Scale the final output frames by this amount. 100%% will output the frames at source dimensions. 50%% at half size 200%% at double size Sets the number of iterations before saving a backup snapshot of the model in it's current state. Set to 0 for off. Sets the number of iterations between each model save. Show training preview output. in a separate window. Skip frames that already have detected faces in the alignments file Skip saving the detected faces to disk. Just create an alignments file Skips frames that have already been extracted and exist in the alignments file Swap the model. Instead converting from of A -> B, converts B -> A Swap the original faces in a source video/images to your final faces.
Conversion plugins can be configured in the 'Settings' Menu The maximum number of parallel processes for performing conversion. Converting images is system RAM heavy so it is possible to run out of memory if you have a lot of processes and not enough RAM to accommodate them all. Setting this to 0 will use the maximum available. No matter what you set this to, it will never attempt to use more processes than are available on your system. If singleprocess is enabled this setting will be ignored. The number of times to re-feed the detected face into the aligner. Each time the face is re-fed into the aligner the bounding box is adjusted by a small amount. The final landmarks are then averaged from each iteration. Helps to remove 'micro-jitter' but at the cost of slower extraction speed. The more times the face is re-fed into the aligner, the less micro-jitter should occur but the longer extraction will take. The output size of extracted faces. Make sure that the model you intend to train supports your required size. This will only need to be changed for hi-res models. To effectively learn, a random set of images are flipped horizontally. Sometimes it is desirable for this not to occur. Generally this should be left off except for during 'fit training'. Train a model on extracted original (A) and swap (B) faces.
Training models can take a long time. Anything from 24hrs to over a week
Model plugins can be configured in the 'Settings' Menu Use the Tensorflow Mirrored Distrubution Strategy to train on multiple GPUs. Warping is integral to training the Neural Network. This option should only be enabled towards the very end of training to try to bring out more detail. Think of it as 'fine-tuning'. Enabling this option from the beginning is likely to kill a model and lead to terrible results. Warps training faces to closely matched Landmarks from the opposite face-set rather than randomly warping the face. This is the 'dfaker' way of doing warping. When used with --frame-ranges outputs the unchanged frames that are not processed instead of discarding them. Writes the training result to a file. The image will be stored in the root of your FaceSwap folder. [LEGACY] This only needs to be selected if a legacy model is being loaded or if there are multiple models in the model folder augmentation faces model output preview settings timelapse training Project-Id-Version: 
PO-Revision-Date: 2021-03-06 11:29+0300
Language-Team: 
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Generated-By: pygettext.py 1.5
X-Generator: Poedit 2.4.2
Last-Translator: 
Plural-Forms: nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<12 || n%100>14) ? 1 : 2);
Language: ru
 Автоматически сохранять файл выравнивания после указанного кол-ва кадров. По умолчанию файл выравнивания сохраняется только в конце процедуры извлечения. Прим.: При извлечении в 2 прохода, файл выравниваний начнёт сохранение только во время второго прохода. ВНИМАНИЕ: Не прерывайте выполнение во время записи, так как это может повлеч порчу файла. Установите в 0 для выключения Размер партии. Это количество изображений для каждой стороны, которые обрабатываются моделью за одну итерацию. Примечание: Поскольку в модель передаётся сразу две стороны за раз, реальное количество загружаемых изображений в два раза больше этого числа. Увеличение размера партии требует больше памяти GPU. Цветовая аугментация помогает модели быть менее чувствительной к разнице цвета между наброами A and B ценой некоторого замедления скорости тренировки. Включите эту опцию для выключения цветовой аугментации. УСТАРЕЛО - Эта настройка будет удалена в будущих обновлениях. Путь к файлу выравнивания для обучающего набора A. По умолчанию используется <input-A> /alignments.json, если он не указан. УСТАРЕЛО - Эта настройка будет удалена в будущих обновлениях. Путь к файлу выравнивания для обучающего набора B. По умолчанию используется <input-B> /alignments.json, если он не указан. Данные Отключить многопроцессорность. Медленнее, но менее ресурсоемко. Отключает журнал TensorBoard. Примечание: Отключение журналов означает, что вы не сможете использовать графики или анализ сессии внутри GUI. Не проводить параллельное извлечение. Вместо одновременного запуска, каждая стадия извлечения будет запущена отдельно (одна, за другой). Полезно при нехватке VRAM. Рисовать лэндмарки на выходных лицах для нужд отладки. Включить преобразование на лету. НЕ рекомендуется. Вам стоит создать чистый файл выравнивания для вашего целевого видео. Однако, если вы хотите, вы можете сгенерировать выравнивания на лету, включив эту опцию. Это приведет к использованию ухудшенного конвеера экстракции и некачественных результатов. Если файл выравниваний найден, этот параметр будет проигнорирован. Обрабатывать каждые N кадров. Эта опция будет пропускать лица при извлечении. Например, значение 1 будет искать лица в каждом кадре, а значение 10 в каждом 10том кадре. Извлеч лица из изображений или видео источников.
Плагины извлечения можно настроить в меню 'Настройки' Обработка лиц Отбрасывает лица ниже указанного размера. Длина указывается в пикселях по диагонали. Установите в 0 для отключения Толькр при использовании файлов nfilter/filter.  Порог для распознавания лица. Чем ниже значения, тем строже. Прим.: Использование фильтра лиц существенно замедлит скорость извлечения. Также точность не гарантируется. Обработка кадров Диапазон кадров к которым применять перенос, например, для кадров от 10 до 50, и 90 до 100 укажите: --frame-ranges 10-50 90-100. Кадры попадающие вне выбранного диапазона будут отброшенны если не указанно '-k' (--keep-unchanged). Прим.: Если при конверсии используются изображения, то имена файлов должны заканчиваться номером кадра! Общие настройки Если лицо не найдено, поворачивает картинку, чтобы попытаться найти лицо. Может найти больше лиц ценой скорости извлечения. Укажите число, чтобы использовать приращения этого размера до 360, либо передайте список чисел, чтобы точно указать, какие углы проверять. Если вы не вычистили ваш файл выравниваний, то вы можете отфильтровать лица указав здесь папку, которая содержит лица извлеченные из входных файлов/видео. Если эта папка указанна, то, только лица, которые существуют в файле выравниваний и ТАКЖЕ существуют в указанной папке будут сконвертированны. Если оставить это поле пустым, то все лица, которые существуют в файле выравниваний будут сконвертированы. Входная папка либо видео файл. Папка с набором фотографий для обработки либо видео файл. Примечание: должно указывать на исходное видео либо набор извлечённых кадров, а НЕ уже извлечённых лица. Входная папка. Папка содержащая изображения для тренировки лица A. Это исходное лицо т.е. лицо, которое вы хотите убрать, заменив лицом B. Входная папка. Папка содержащая изображения для тренировки лица B. Это новое лицо т.е. лицо, которое вы хотите поместить на голову человека A. Кол-во итераций для тренировки. Используется только для автоматизирования. Не существует "правильного" кол-ва итераций для любой выбранно модели. Тренировку стоит завершать только когда вы довольны кадрами на превью. Однако, если вы хотите, чтобы тренировка прервалась после указанного кол-ва итерация, вы можете ввести это здесь. Уровень записи журнала. Придерживайтесь уровней INFO или VERBOSE, кроме случаев когда вам нужно отправить отчёт об ошибке. Будьте осторожнее при указании уровня TRACE, так как будет сгенерированно очень много данных Папка с моделью. Папка, содержащая обученную модель, которую вы хотите использовать для преобразования. Папка сохранений модели. Здесь сохраняется прогресс тренировки. Следует всегда создавать новую папку для новых моделей. При начале тренировки новой модели, выберите пустую либо несуществующую папку (во втором случае она будет создана). Если вы хотите продолжить тренировку, выберите папку с уже существующими сохранениями. Нужно указывать лишь при конвертации из набора картинок в видео. Предоставьте исходное видео, из которого были извлеченны кадры (для настройки частоты кадров, а также аудио). Опционально, при создании таймлапса. Создаст картинку текущего таймлапса выбранных лиц в папке timelapse-output при каждом сохранении модели. Если указанны только входные папки, то по умолчанию вывод будет сохранён вместе с моделью в подкаталог /timelapse/ Только при создании таймлапсов. Сохраняет предварительный просмотр выбранных лиц в папку timelapse-output при каждом сохранении. Следует указать входную папку лиц набора 'A' для использования при создании таймлапса. Вам также нужно указать параметры--timelapse-output и --timelapse-input-B. Только при создании таймлапса. Таймлапс будет сохранять изображения выбранных лиц в папке таймлапсов при каждой итерации сохранения. Это должна быть папка для ввода лиц из набора 'B', для использования в создании таймлапса. Вы также должны указать параметр --timelapse-output и --timelapse-input-A. Путь к файлу выравнивания. Оставьте пустым, для пути  по муолчанию. Дополнительно вы можете отфильтровать лица людей, которых вы не хотите обрабатывать указав изображение этого человека. На изображении должен быть фронтальный портрет одного человека . Можно указать несколько файлов через пробел. Прим.: Фильтрация лиц существенно снижает скорость извлечения, при этом точность не гарантируется. Переобозначить путь к файлу конфигурации пользовательским. (Необязательно) Доролнительно вы можете выбрать людей, которых выхотели бы включить в обработку путём указания изображения этого человека. Должен быть фронтальный портрет с лишь одним человеком на картинке. Можно выбрать несколько изображений через пробел. Прим.: Использвание фильтра существенно замедлит скорость извлечения. Также точность не гарантируется. Папка для сохранения преобразованных файлов. Вывод в системную консоль вместо GUI Путь для сохранения файла журнала. Оставьте пустым, чтобы сохранить в папке с faceswap Величина в процентах, на которую требуется масштабировать предварительный просмотр. 100 %% - размер вывода модели. Плагины R|Создание доп. масок. Генерация масок требует дополнительной памяти GPU. Вы можете выбрать none, одну, или несколько масок, но процес извлечение может занять больше времени в зависимости от выбора. Прим.: Маски Extended и Components (на основе меток лица) всегда создаются автоматически при извлечении лиц.
L|vgg-clear: Маска предназначена для умной сегментации преимущественно фронтальных лиц без препятствий. Фотографии в профиль могут быть обработанны посредственно.
L|vgg-obstructed: Маска предназначена для умной сегментации преимущественно фронтальных лиц. Эта маска была обучена распозновать некоторые препятствия, такие как руки и очки. Фотографии в профиль могут быть обработанны посредственно.
L|unet-dfl: Маска предназначена для умной сегментации преимущественно фронтальных лиц. Маска была обучена силами участников сообщества и нуждается в тестировании. Фотографии в профиль могут быть обработанны посредственно.
Следующие маски создаются автоматически:
L|components: Маска предназначена для сегментации лица на основе ориентиров лица. Маска создается путом построения выпуклого полигона вокруг внешних ориентиров лица.
L|extended: Маска предназначена для сегментации лица на основе ориентиров лица. Маска создается путом построения выпуклого полигона вокруг внешних ориентиров лица и расширяется вверх на лоб.
(пример: `-M unet-dfl vgg-clear`, `--masker vgg-obstructed`) R|Выравнивание лица.
L|cv2-dnn: Детектор меток лица, только для CPU. Быстрый, не требователен к ресурсам, но менее точный. Используйте только если вам небходимо не использовать GPU.
L|fan: Лучший выравниватель. Быстрый на GPU, медленный на CPU. R|Тип детектора. Некоторые могут быть настроенны через '/config/extract.ini' либо 'Settings > Configure Extract 'Plugins':
L|cv2-dnn: Работает только на CPU, наименее надежный и наименее требователен к ресурсам. Используйте если для вас очень важна скорость, а также не использовать GPU .
L|mtcnn: Хороший детектор. Быстрый на CPU, ещё быстрее на GPU. Использует меньше ресурсов, нежели другие GPU детекторы, но может производить больше ложных положительных детектирований.
L|s3fd: Лучший детектор. Медленный на CPU, быстре на GPU. Может детектировать лицо в большем кол-ве ситуация и меньшим кол-вом ошибок, чем другие GPU, но значительно более требователен к ресурсам. R|Не использовать GPU для Faceswap. Выберите номер(а), которые соответствуют тем GPU, которые вы не хотите использовать в Faceswap. При отключении всех GPU Faceswap будет работать в режиме CPU. R|Использовать маску. Прим.: Требуемая маска должна наличествовать в файле выравнивания. Доп. маски можно добавить через Инструмент Создания Масок.
L|none: Не использовать маску.
L | компоненты: маска, предназначенная для сегментации лица на основе найденных ориентиров. Маска создается построением выпуклого многоугольника вокруг внешних ориентиров лица.
L | расширенный: маска, предназначенная для сегментации лица на основе расположения ориентиров. Маска создается построением выпуклого многоугольника вокруг внешних ориентиров лица и продолжается вверх на лоб.
L | vgg-clear: маска, предназначенная для умной сегментации преимущественно фронтальных лиц без препятствий. Лица в профиль и препятствия могут привести к некачественным результатам.
L | vgg-obstructed: маска, предназначенная для умной сегментации преимущественно фронтальных лиц. Модель маски специально обучена распознавать некоторые лицевые препятствия (руки и очки). Лица в профиль могут привести к некачественным результатам..
L | unet-dfl: маска, предназначенная для умной сегментации преимущественно фронтальных лиц. Модель маски была обучена членами сообщества и потребует тестирования для дальнейшего описания. Лица в профиль могут привести к некачественным результатам.. R|Нормализация может помоч выравниванию лиц при сложных условиях освещения, ценой снижения скорости. Различные методы дают разные результаты в зависимости от набора лиц. Прим.: Не влияет на вывод лица, только на выравнивание.
L|none: Не производить нармализацию картинки лица.
L|clahe: Производить нормализацию методом CLAHE.
L|hist: Выравнивание гистрограммы каналов RGB каналов.
L|mean: Усреднение цветов лица. R|Производит подгонку цветов в изменённом лице. Некоторые из этих опций имеют настройки в  файле '/config/convert.ini' либо 'Настройки > Настроить Плагины Конверсии':
L|avg-color: Подогнать среднее значение каждого цветового канала в замененном лице так, чтобы оно равнялось среднему значению области маски исходного изображения.
L|color-transfer: Переносит распределение цвета от источника к целевому изображению с использованием среднего и стандартного отклонения цветового пространства L * a * b *.
L|manual-balance: Ручная настройка баланса изображения в различных цветовых пространствах. Лучше всего использовать с инструментом предварительного просмотра для установки правильных значений.
L|match-hist: Подгонять гистограмму каждого цветового канала нового лица, гистограммой области маски исходного изображения
L|seamless-clone: Исп. фунцю cv2's незаметного переноса чтобы убрать экстремальные градиенты на краях маски путём сглаживания цветов. Обычно не даёт удовотворительных результатов.
L|none: Не прозиводить подгонку цвета. R|Выберите тренера для использования. Тренеры могут быть настроенны через меню Настройки либо в папке config.
L|original: Оригинальная модель созданная /u/deepfakes.
L|dfaker: модель с 64px вход/128px выходом от dfaker. Включите 'warp-to-landmarks' для полно соотвествия методу dfaker.
L|dfl-h128: 128px вход/вызод модель от deepfacelab
L|dfl-sae: Адаптивная модель от deepfacelab
L|dlight: Легковесная модель высокого разрешения. Один из вариантов DFaker.
L|iae: Модель использующая промежуточные слои, для достижения лучшей детализции
L|lightweight: Легковесная модель для младшей линейки видеокарт. Не ожидайте хороших результатов. Может тренировать на картах с 1.6Гб памяти при размере серии 8.
L|realface: Модель повышенной детализации, с двумя сложносоставными слоями, базированная на DFaker, с настраеваемым разрешением входа/выхода. Автоэнкодеры несбалансированны, поэтому свапы B>A  не дадут хорошего качества. andenixa и другие. Очень настраеваемая.
L|unbalanced: Модель 128px вход/выход от andenixa. Автоэнкодеры несбалансированы, поэтому свапы B>A не будут очень хорошими. Очень настраеваемая.
L|villain: Модель 128px вход/выход от villainguy. Очень трбовательна к ресурсам (Вам потребуется GPU с хороши количеством видеопамяти). Хороша для деталей, но подверженна к неправильной передаче цвета. R|Тип плагина для вывода сконвертированных изображений. Записывающие плагины можно настроить в '/config/convert.ini' либо 'Настройки > Настроить Плагины Конверсии:'
L|ffmpeg: [видео] Записывает результат конверсии сразу в видео файл. Если входом является серий изображений, то нужно такаже указать параметер '-ref' (--reference-video).
L|gif: [анимированное изображение] Создает анимированный gif.
L|opencv: [изображения] Наибыстрейший способ записи, но с меньшим кол-вом опций и форматов вывода.
L|pillow: [изображения] Более медленный, чем opencv, но имеет больше опций и поддерживает больше форматов. Сохранение Маштабировать оконечные кадры до указанного процента. 100%% будет выводить кадры в исходном размере. 50%% половина от размера, а 200%% в удвоенном размере Устанавливает кол-во итераций перед созданием резервной копии модели. Установите в 0 для отключения. Установка количества итераций между сохранениями модели. Показывать предварительный просмотр в отдельном окне. Пропускать кадры, для которых в файле выравнивания есть найденные лица Не сохранять найденные лица на носитель. Просто создать файл выравнивания Пропускать кадры, которые уже были извлечены и существуют в файле выравнивания Поменять модели местами. Вместо преобразования из A -> B, преобразует B -> A Заменить  оригиналы лица в исходном видео/фотографиях новыми.
Плагины конвертации могут быть настроенны в меню 'Настройки' Максимальное количество параллельных процессов для выполнения преобразования. Преобразование изображений требует большого объема системной памяти, поэтому возможна её нехватка, если у вас много процессов и не хватает памяти для их всех. Установка этого значения на 0 будет использовать максимально доступное значение. Независимо от ваших установок, никогда не будет использоваться больше процессов, чем доступно в вашей системе. Если включен одиночный процесс, этот параметр будет проигнорирован. Кол-во проходов выравнивания после обнаружения лица. Каждый раз при повторном выравнивании рамка лица немного корректируется. Окончательные ориентиры затем усредняются. Помогает устранить «микроджиттер», но за счет замедления скорости извлечения. Чем больше проходов выравнивания, тем меньше микродрожание, но тем дольше идёт извлечение. Размер извлекаемых лиц в пикселях. Убедитесь, что выбранная Вами модель поддерживает такой входной размер. Стоит изменять только для моделей высокого разрешения. Для повышения эффективности обучения, некоторые изображения случайным образом переворачивается по горизонтали. Иногда желательно, чтобы этого не происходило. Как правило, эту настройку не стоит трогать, за исключением периода «финальной шлифовки». Начать обучение модели используя наборы лиц: (A) - исходное лицо и (B) - новое лицо.
Обучение моделей может занять долгое время: от 24 часов до недели
Каждую модель можно отдельно настроить в меню «Настройки» Использовать стратегию зеркального распределения Tensorflow для совместной тренировки сразу на нескольких GPU. Внсени случайных искажение является неотъемлемой частью обучения нейронной сети. Эту опцию следует включать только в самом конце обучения, чтобы попытаться выявить больше деталей. Думайте об этом как о «стадии шлифовки». Включение этой опции с самого начала может убить модель и привести к ужасным результатам. Вместо случайного искажения лица, деформирует лица в соотвествии с Ориентирами/Landmarks противоположного набора лиц. Этот способ используется пакетом "dfaker". При использовании с --frame-range кадры не попавшие в диапазон выводятся неизмененными, вместо их пропуска. Записывает результат тренировки в файл. Файл будет сохранён в коренной папке FaceSwap. [СОВМЕСТИМОСТЬ] Это нужно выбирать только в том случае, если загружается устаревшая модель или если в папке сохранения есть несколько моделей аугментация лица модель вывод предварительный просмотр настройки таймлапс тренировка 