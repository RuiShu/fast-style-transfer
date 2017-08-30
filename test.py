from config import args
import tensorbayes as tb
import tensorflow as tf
from utils import get_img_files, get_img, save_img
import numpy as np
import os
import subprocess

def execute(exec_str):
    print "EXEC:\n\t{:s}".format(exec_str)
    subprocess.call(exec_str, shell=True)

def get_model_path():
    if args.ckpt:
        path = os.path.join(args.model_dir, 'model-{:d}'.format(args.ckpt))
    else:
        path = tf.train.latest_checkpoint(args.model_dir)
    return path

def evaluate(M, test_files, out_dir, batch_size=1, prefix=''):
    f_batches = list(tb.nputils.split(test_files, batch_size))
    i_total = len(f_batches)

    for i, f_batch in enumerate(f_batches):
        batch = np.array([get_img(f) for f in f_batch])
        imgs = M.sess.run(M.y_test, {M.x_test: batch})

        tb.utils.progbar(i, i_total)

        for f, img in zip(f_batch, imgs):
            if prefix != '':
                img_path = prefix + '_' + os.path.basename(f)
            else:
                img_path = os.path.basename(f)
            path = os.path.join(out_dir, img_path)
            save_img(path, img)

def test(M):
    with M.graph.as_default():
        saver = tf.train.Saver()
        saver.restore(M.sess, get_model_path())
        print "Restored from {:s}".format(get_model_path())

    if args.test_dir:
        test_files = get_img_files(args.test_dir)
        evaluate(M, test_files, 'results',
                 batch_size=1,
                 prefix=args.model_name)

    if args.test_file:
        if args.test_file.endswith(('.png', '.jpg')):
            test_files = [args.test_file]
            evaluate(M, test_files, 'results',
                     batch_size=1,
                     prefix=args.model_name)

        elif args.test_file.endswith('.mp4'):
            in_video_name = os.path.basename(args.test_file).rstrip('.mp4')
            out_video_name = in_video_name + '_' + args.model_name
            test_dir = 'results/{:s}/in'.format(in_video_name)
            out_dir = 'results/{:s}/{:s}'.format(in_video_name, args.model_name)

            try:
                os.makedirs(test_dir)
                exec_str = ("ffmpeg -y -i {:s} {:s}/frame_%d.png"
                            .format(args.test_file, test_dir))
                print "EXEC:\n\t{:s}".format(exec_str)
                subprocess.call(exec_str, shell=True)
            except OSError:
                print "Using existing {:s}".format(test_dir)

            try:
                os.makedirs(out_dir)
                test_files = get_img_files(test_dir)
                evaluate(M, test_files, out_dir,
                         batch_size=20)
            except OSError:
                print "Using existing {:s}".format(out_dir)

            out_video_path = 'results/{:s}.mp4'.format(out_video_name)
            out_audio_video_path = 'results/{:s}_audio.mp4'.format(out_video_name)
            execute("ffmpeg -y -r 23.98 -f image2 -i {:s}/frame_%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {:s}"
                    .format(out_dir, out_video_path))
            execute("ffmpeg -y -i {:s} -i {:s} -c:v copy -c:a copy {:s}"
                    .format(out_video_path, args.test_file, out_audio_video_path))
            execute("mv {:s} {:s}".format(out_audio_video_path, out_video_path))
