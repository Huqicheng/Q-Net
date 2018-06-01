import numpy as np
from gen_data import CAPTCHA_LIST,CAPTCHA_LEN, gen_captcha_text_and_image,CAPTCHA_WIDTH,CAPTCHA_HEIGHT
from deeplearning.data import DataIterator, Batch, Iterator

def convert2gray(img):
    '''
        :param img:
        :return:
    '''
    if len(img.shape) > 2:
        img = np.mean(img, -1)
    return img


def text2vec(text, captcha_len=CAPTCHA_LEN, captcha_list=CAPTCHA_LIST):
    '''
        :param text:
        :param captcha_len:
        :param captcha_list:
        :return:
    '''
    text_len = len(text)
    if text_len > captcha_len:
        raise ValueError('No more than 4 characters.')
    vector = np.zeros(captcha_len * len(captcha_list))
    for i in range(text_len):
        vector[captcha_list.index(text[i])+i*len(captcha_list)] = 1
    return vector


def vec2text(vec, captcha_list=CAPTCHA_LIST, size=CAPTCHA_LEN):
    '''
        :param vec:
        :param captcha_list:
        :param size:
        :return:
    '''
    vec_idx = vec
    text_list = [captcha_list[v] for v in vec_idx]
    return ''.join(text_list)


def wrap_gen_captcha_text_and_image(shape=(60, 160, 3)):
    '''
        :param shape:
        :return:
    '''
    while True:
        t, im = gen_captcha_text_and_image()
        if im.shape == shape: return t, im



def next_batch(batch_count=60, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    '''
        :param batch_count:
        :param width:
        :param height:
        :return:
    '''
    batch_x = np.zeros([batch_count, width * height])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])
    for i in range(batch_count):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)
                         
    return batch_x, batch_y


class CaptchaBatchIterator:
    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size
    
    def __call__(self, batch_num) -> Iterator[Batch]:
        for i in range(batch_num):
            batch_inputs, batch_targets = next_batch(self.batch_size)
            batch_inputs = batch_inputs.reshape((self.batch_size,1,CAPTCHA_HEIGHT,CAPTCHA_WIDTH))
            yield Batch(batch_inputs, batch_targets)











