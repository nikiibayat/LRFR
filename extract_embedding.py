# The VGG-face trained network is used to produce a feature vector for the HR image and all of the SR images.
# https://github.com/rcmalli/keras-vggface
import numpy as np
from keras.preprocessing import image
from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils


def predict(img_path, classify=False):
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=2)
    preds = model.predict(x)
    if classify:
        # print('Predicted:', utils.decode_predictions(preds))
        return utils.decode_predictions(preds)
    else:
        # print("Features: ", preds.shape)
        return preds


def extract_features(img_path):
    vgg_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max
    return predict(vgg_features, img_path, classify=False)
    # if you want to extract features of a specific layer
    # layer_name = 'layer_name' # edit this line
    # vgg_model = VGGFace() # pooling: None, avg or max
    # out = vgg_model.get_layer(layer_name).output
    # vgg_model_new = Model(vgg_model.input, out)


def main():
    # vggface = VGGFace(model='resnet50') # models: vgg16, resnet50, senet50
    predict("./SRGAN.png", classify=False)


if __name__ == "__main__":
    main()

