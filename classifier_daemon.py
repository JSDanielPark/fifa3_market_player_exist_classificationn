import tensorflow as tf
import pyscreenshot as imgrab
from scipy import misc
import numpy as np
from net import Net
from common import input_x, input_y, input_size
import time
import requests

# SMS 발송용 정보 (본인은 문자나라 이용)
userid = ""
passwd = ""
sender_phone_number = ""
receiver_phone_number = ""
message = "FifaOnline3 Player Registed!"
sms_url = "http://211.233.20.184/MSG/send/web_admin_send.htm?userid="\
          + userid \
          + "&passwd=" \
          + passwd \
          + "&sender=" \
          + sender_phone_number \
          + "&receiver=" \
          + receiver_phone_number\
          +"&message=" \
          + message

if __name__ == '__main__':
    with tf.Session() as sess:
        net = Net(sess)
        net.restore()

        while True:
            im = imgrab.grab(bbox=(0, 0, 360, 720))
            im = misc.imresize(im, (input_x, input_y))
            im = np.reshape(im, [input_size])

            if int(net.predict(im, 1.0)[0][0]) == 1:
                requests.get(sms_url)
            time.sleep(5)
