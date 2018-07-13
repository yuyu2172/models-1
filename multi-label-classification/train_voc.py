import argparse

import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import VOCBboxDataset

from chainercv.datasets import voc_bbox_label_names

from chainercv.chainer_experimental.datasets.sliceable import\
    TransformDataset

from lib.get_resnet import get_resnet_50
from lib.multi_label_classifier import MultiLabelClassifier

from lib.transform import bbox_to_multi_label


def converter(batch, device=None):
    # do not send data to gpu (device is ignored)
    return tuple(list(v) for v in zip(*batch))


def main():
    parser = argparse.ArgumentParser(
        description='Chainer Multi-label classification')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', type=int, default=4,
                        help='Number of images in each mini-batch')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    model = get_resnet_50(len(voc_bbox_label_names))
    model.pick = 'fc6'
    train_chain = MultiLabelClassifier(
        model, loss_scale=len(voc_bbox_label_names))

    train = VOCBboxDataset(
        year='2007', split='trainval', use_difficult=False)
    train = TransformDataset(
        train, ('img', 'bbox'), bbox_to_multi_label)
    test = VOCBboxDataset(
        year='2007', split='test', use_difficult=False)
    test = TransformDataset(
        test, ('img', 'bbox'), bbox_to_multi_label)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        train_chain.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(0.001)
    optimizer.setup(train_chain)

    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(1e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    stop_trigger = (11, 'epoch')
    log_interval = (20, 'iteration')

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu, converter=converter)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)
    trainer.extend(
        extensions.Evaluator(
            test_iter, train_chain, device=args.gpu, converter=converter))
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1),
        trigger=triggers.ManualScheduleTrigger(
            [8, 10], 'epoch'))

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['lr',
         'epoch',
         'elapsed_time',
         'main/loss',
         'main/recall',
         'main/precision',
         'main/n_pred',
         'main/n_pos',
         'validation/main/loss',
         'validation/main/recall',
         'validation/main/precision',
         'validation/main/n_pred',
         'validation/main/n_pos',
         ]), trigger=log_interval)

    trainer.extend(
        extensions.snapshot_object(model, 'snapshot_model.npz'))
    trainer.extend(extensions.LogReport(trigger=log_interval))

    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == '__main__':
    main()
