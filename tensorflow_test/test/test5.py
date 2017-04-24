import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("flower_photos/daisy/5547758_eea9edfd54_n.jpg",'r').read()

sess = tf.Session()
with sess.as_default():
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())

    plt.imshow(img_data.eval())
    plt.show()

    image_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)

    encode_img = tf.image.encode_jpeg(image_data)
    with tf.gfile.GFile("./output","wb") as f:
        f.write(encode_img.eval())
