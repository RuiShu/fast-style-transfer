from config import args
from utils import delete_existing, get_img, get_img_files
import tensorbayes as tb
import tensorflow as tf
import numpy as np
import os

def push_to_buffer(buf, data_files):
    files = np.random.choice(data_files, len(buf), replace=False)
    for i, f in enumerate(files):
        buf[i] = get_img(f, (256, 256, 3))

def train(M):
    delete_existing(args.log_dir)
    train_writer = tf.summary.FileWriter(args.log_dir)
    train_files = get_img_files(args.train_dir)
    validation_files = get_img_files(args.validation_dir)
    iterep = args.iter_visualize

    with M.graph.as_default():
        M.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

    batch = np.zeros((args.batch_size, 256, 256, 3), dtype='float32')

    for i in xrange(len(train_files) * args.n_epochs):
        push_to_buffer(batch, train_files)
        summary, _ = M.sess.run(M.ops_main, {M.x: batch})
        train_writer.add_summary(summary, i + 1)
        train_writer.flush()

        message='i={:d}'.format(i + 1)
        end_viz, _ = tb.utils.progbar(i, iterep, message)

        if (i + 1) % args.iter_visualize == 0:
            for f, op in zip(validation_files, M.ops_images):
                img = np.expand_dims(get_img(f), 0)
                summary = M.sess.run(op, {M.x_test: img})
                train_writer.add_summary(summary, i + 1)

        if (i + 1) % args.iter_save == 0:
            path = saver.save(M.sess, os.path.join(args.model_dir, 'model'),
                              global_step=i + 1)
            print "Saving model to {:s}".format(path)
