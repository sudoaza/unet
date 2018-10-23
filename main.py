from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

project = 'objects'

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

train_path = f"data/{project}/train"
myGene = trainGenerator(2, train_path, 'image','label', data_gen_args, save_to_dir = None)

save_path = f"unet_{project}.hdf5"
model = unet()
model_checkpoint = ModelCheckpoint(save_path, monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=100, epochs=1, callbacks=[model_checkpoint])

test_path = f"data/{project}/test"
testGene = testGenerator(test_path)
results = model.predict_generator(testGene, 30, verbose=1)
saveResult(test_path, results)
