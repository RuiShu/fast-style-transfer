from config import args
from utils import get_img, get_img_files, delete_existing
from .vgg import vgg_features
from .fast_style import fast_style
import tensorflow as tf
import tensorbayes as tb
import numpy as np
from tensorbayes.layers import placeholder

class Model(object):
    def __init__(self):
        with tf.Graph().as_default() as graph, tf.Session() as sess:
            self.x = placeholder((None, 256, 256, 3), name='x')
            self.graph = graph
            self.sess = sess
            # During test time, spatial dims are unknown
            self.x_test = placeholder((None, None, None, 3), name='x_test')

        self.build()

    def build(self):
        print "Building style-transfer model"
        with self.graph.as_default():
            self.y_test = fast_style(self.x_test / 255.)

        if args.train:
            self.training_build()

    def training_build(self):
        print "Pre-computing features"
        with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
            img = np.expand_dims(get_img(args.target_style), 0)
            x = tb.layers.placeholder(img.shape, name='x')
            target_styles, _ = vgg_features(x, get_style=True)
            target_styles = sess.run(target_styles, {x: img})

        with self.graph.as_default():
            y = fast_style(self.x / 255., reuse=True)
            _, target_contents = vgg_features(self.x, get_content=True)
            styles, contents = vgg_features(y, get_style=True, get_content=True)

            print "Building training loss"
            with tf.name_scope('content_loss'):
                content_losses = []
                for layer in contents:
                    size = np.prod(contents[layer]._shape_as_list()[1:])
                    loss = tf.nn.l2_loss(target_contents[layer] - contents[layer])
                    content_losses += [2 * loss / size]

                content_loss = tf.add_n(content_losses) / args.batch_size

            with tf.name_scope('style_loss'):
                style_losses = []
                for layer in styles:
                    # Get size of 2nd moment matrix
                    size = target_styles[layer].size
                    loss = tf.nn.l2_loss(target_styles[layer] - styles[layer])
                    style_losses += [2 * loss / size]

                style_loss = tf.add_n(style_losses) / args.batch_size

            with tf.name_scope('loss'):
                loss = (args.content_weight * content_loss +
                        args.style_weight * style_loss)

            # Optimizer
            with tf.variable_scope('gradients'):
                train_main = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)

            # Summaries
            tf.summary.scalar('main/content_loss', content_loss)
            tf.summary.scalar('main/style_loss', style_loss)
            tf.summary.scalar('main/total_loss', loss)
            summary_main = tf.summary.merge(tf.get_collection('summaries', 'main'))

            summary_images = []
            for i in xrange(len(get_img_files(args.validation_dir))):
                summary = tf.summary.image('image/image_{:d}'.format(i), self.y_test / 255.)
                summary_images.append(summary)

            # Cache relevant ops
            self.ops_main = [summary_main, train_main]
            self.ops_images = summary_images
