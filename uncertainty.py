import numpy as np
import customize_obj
import tensorflow as tf
from deoxys.experiment import ExperimentPipeline, model_from_full_config
import argparse
import h5py



if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("log_folder")
    parser.add_argument("output_file")
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--analysis_folder",
                        default='', type=str)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--model_checkpoint_period", default=5, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=5, type=int)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument("--monitor", default='', type=str)
    parser.add_argument("--best_epoch", default=0, type=int)
    parser.add_argument("--memory_limit", default=0, type=int)

    args, unknown = parser.parse_known_args()

    # ex = ExperimentPipeline(
    #     log_base_path=args.log_folder,
    #     temp_base_path=args.temp_folder
    # )

    # if args.best_epoch == 0:
    #     try:
    #         ex = ex.load_best_model(
    #             monitor=args.monitor,
    #             recipe='auto',
    #             map_meta_data=args.meta,
    #         )
    #     except Exception as e:
    #         print(e)
    # else:
    #     print(f'Loading model from epoch {args.best_epoch}')
    #     ex.from_file(args.log_folder +
    #                  f'/model/model.{args.best_epoch:03d}.h5')

    @tf.custom_gradient
    def _GuidedBackPropRelu(x):
        res = tf.nn.relu(x)

        def grad(dy):
            return dy * tf.cast(dy > 0., x.dtype) * tf.cast(x > 0., x.dtype)

        return res, grad
    print('Loading model from ', args.config_file, ' with weights from model ', args.log_folder, ' epoch ', args.best_epoch)
    mc_model = model_from_full_config(args.config_file, weights_file=args.log_folder +
                     f'/model/model.{args.best_epoch:03d}.h5')

    for layer in mc_model.model.layers:
        if 'activation' in layer.get_config():
            if 'relu' in layer.activation.__name__:
                layer.activation = _GuidedBackPropRelu

    predict_fn = tf.function(mc_model.model)
    # predict_fn = tf.function(tf.keras.models.Model(mc_model.model.inputs, mc_model.model.output))

    last_layer = mc_model.model.layers[-1].name

    # go through test data
    dr = mc_model.data_reader
    test_gen = dr.test_generator.generate()
    BATCH_SIZE = 1

    print('initializing output files')
    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset('mc_output',
                        data=np.zeros((40, 173, 191, 265, 1)),
                        chunks=(1, 173, 191, 265, 1), compression='gzip')
        f.create_dataset('gb_output', data=np.zeros((40, 173, 191, 265, 2)),
                        chunks=(1, 173, 191, 265, 2), compression='gzip')

    print('calculating outputs')
    for i, (images, target) in enumerate(test_gen):
        print('calculating for batch', i)
        img_tensor = tf.constant(images)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            mc_output = predict_fn(img_tensor)
        gb_output = tape.gradient(mc_output, img_tensor)

        # gb_output = mc_model.guided_backprop(last_layer, images)

        start, end = i*BATCH_SIZE, (i+1) * BATCH_SIZE

        print('saving to file....')
        with h5py.File(args.output_file, 'a') as f:
            f['mc_output'][start:end] = mc_output.numpy()
            f['gb_output'][start:end] = gb_output.numpy()

        if i >= 40:
            break
