import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import argparse

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    # ********************* DEFAULT INPUT VARIABLES (edit if necesary) *************************
    model2load = 'fcnncifar'
    models_dir = 'pretrainedmodels/'
    output_dir = 'results/'
    figs_dir = 'figures/'
    data_dir = 'datasets/'
    n_images = 2000
    max_epsilon = 0.2
    min_epsilon = 0.05
    # ********************* ******************************************* *************************
    '''
    For cifar is good: 
    python main.py --lr=0.01 --n-images=2000 --bsize=256 --start-advtrain=0.3 --adv-acc-thrs=0.1 --min-epsilon=0.001 --max-epsilon=0.2 --epochs=50 --its-advtrain=1
    python main.py --lr=0.01 --n-images=2000 --bsize=256 --start-advtrain=0.3 --adv-acc-thrs=0.4 --min-epsilon=0.0001 --max-epsilon=0.2 --epochs=10 --its-advtrain=1
    '''


    parser = argparse.ArgumentParser(description="Benchmarking of Algorithms for Generation of Adversarial Examples")
    parser.add_argument("--rho-mode", help="Robustness Measure Mode: compute/store the robustness measures and exit",
                        action="store_true")
    parser.add_argument("--model2load", type=str, default=model2load,
                        help="model to be loaded: either of these --> fcnn, lenet, nin, densenet. Default value = " + model2load)
    parser.add_argument("--models-dir", type=str, default=models_dir,
                        help="Path to the directory containing the pre-trained model(s). Default value = " + models_dir)
    parser.add_argument("--output-dir", type=str, default=output_dir,
                        help="Path to the directory where the output fooling ratio(s) will be stored. Default value = " + output_dir)
    parser.add_argument("--figs-dir", type=str, default=figs_dir,
                        help="Path to the directory where the output figure(s) will be stored. Default value = " + figs_dir)
    parser.add_argument("--data-dir", type=str, default=data_dir,
                        help="Path to the directory containing the dataset(s). Default value = " + data_dir)
    parser.add_argument("--n-images", type=int, default=n_images,
                        help="Number of images of the dataset to be fooled. Default value = " + str(n_images))
    parser.add_argument("--max-epsilon", type=float, default=max_epsilon)
    parser.add_argument("--min-epsilon", type=float, default=min_epsilon)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--adv-acc-thrs", type=float, default=0.9)
    parser.add_argument("--start-advtrain", type=float, default=0.5)
    parser.add_argument("--its-advtrain", type=int, default=10)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=100)
    return parser.parse_args()

def get_all_model_variables(args):
    '''
    Every model has its own names for the tensors in the grahp.
    After looking at the graph of each model, in tensorboard using 'ViewGraph.py',
    we create a dictionary with the tensors names of:
        (*) 'input': 	The input image of the model
        (*) 'logits': 	The output vector of the last layer of the model (before softmax is applied).
        (*)	'pkeep':	Dropout probability if applicable, None otherwise
    '''
    if (args.model2load == 'fcnnmnist'):
        modelvarnames = {
        'model2load':args.model2load,
        'models_dir':args.models_dir,
        'output_dir':args.output_dir,
        'figs_dir':args.figs_dir,
        'data_dir':args.data_dir,
        'n_images':args.n_images,
        'max_epsilon':args.max_epsilon,
        'min_epsilon':args.min_epsilon,
        'graph_directory':'savedmodel_fcnn_mnist/',
        'graph_file':'fcnn.ckpt.meta',
        'input':'x:0',
        'logits':'logits:0',
        'lr':args.lr,
        'batch-size':args.bsize,
        'adv-acc-thrs':args.adv_acc_thrs,
        'start-advtrain':args.start_advtrain,
        'its-advtrain':args.its_advtrain,
        'epochs':args.epochs,
        'pkeep':None
        }
    elif (args.model2load == 'fcnncifar'):
        modelvarnames = {
        'model2load':args.model2load,
        'models_dir':args.models_dir,
        'output_dir':args.output_dir,
        'figs_dir':args.figs_dir,
        'data_dir':args.data_dir,
        'n_images':args.n_images,
        'max_epsilon':args.max_epsilon,
        'min_epsilon':args.min_epsilon,
        'graph_directory':'savedmodel_fcnn_cifar/',
        'graph_file':'fcnn_cifar.ckpt.meta',
        'input':'x:0',
        'logits':'logits:0',
        'lr':args.lr,
        'batch-size':args.bsize,
        'adv-acc-thrs':args.adv_acc_thrs,
        'start-advtrain':args.start_advtrain,
        'its-advtrain':args.its_advtrain,
        'epochs':args.epochs,
        'pkeep':None
        }
    else:
        print('Error:  Select a valid model (fcnnmnist, fcnncifar)')
        modelvarnames = None

    return modelvarnames

def pre_process_data(X, y, model2load):
    if model2load == 'fcnnmnist':
        X = np.reshape(X, [X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])),1])
        X = np.pad(X, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        y = np.argmax(y,1)
    elif model2load=='fcnncifar':
        X = np.reshape(X, [-1, 32*32*3])
        y = np.argmax(y, 1)

    X, y = shuffle(X, y)
    return X, y

def predict_CNN(X, modelvarnames):
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config = config) as sess:
        saver = tf.train.import_meta_graph(modelvarnames['models_dir'] +\
                                           modelvarnames['graph_directory'] +\
                                           modelvarnames['graph_file'])
        saver.restore(sess, tf.train.latest_checkpoint(modelvarnames['models_dir'] +\
                                                       modelvarnames['graph_directory']))

        graph = tf.get_default_graph()

        Xinput = graph.get_tensor_by_name(modelvarnames['input'])
        Ylogits = graph.get_tensor_by_name(modelvarnames['logits'])
        if modelvarnames['pkeep'] == None:
            pkeep = tf.placeholder(tf.int32, (None))  # Not used
        else:
            pkeep = graph.get_tensor_by_name(modelvarnames['pkeep'])

        Ysoftmax = tf.nn.softmax(Ylogits)
        predicted_labels = tf.argmax(Ylogits, 1)
        confidence = tf.reduce_max(Ysoftmax, 1)

        Yout, yconf = sess.run([predicted_labels, confidence], feed_dict={Xinput: X, pkeep:1.0})
    return Yout, yconf

def get_adversarial_examples(X, ypred, model, epsilon, iterations=1):

    x = X + np.random.uniform(-epsilon, epsilon, X.shape) # Random Start
    for i in range(iterations):
        grad = model.sess.run(model.grad_loss,
                              feed_dict={model.Xtrue: x,
                                         model.Ytrue: ypred,
                                         model.pkeep:1.0})
        x = np.add(x, epsilon/iterations * np.sign(grad), out=x, casting='unsafe')
        x = np.clip(x, X - epsilon, X + epsilon)
    return x

def fclayer(input, in_len, out_len, activation, name):
    mu = 0
    sigma = 1./np.sqrt(in_len)
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(shape = (in_len, out_len), mean = mu, stddev = sigma), name='W')
        b = tf.Variable(tf.zeros(out_len), name='b')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W)
        out = activation(tf.matmul(input, W) + b)
        tf.identity(out, name='out')
    return out

def fcnn(layers_depth, modelvarnames, activation=tf.nn.relu):
    x = tf.placeholder(tf.float32, [None, layers_depth[0]], name='x')  # input variable
    y = tf.placeholder(tf.int64, [None], name='y')  # predicted label (onehot format)

    # logits = tf.layers.flatten(x)
    logits = x
    for i in range(len(layers_depth) - 1):
        if i == len(layers_depth) - 2:
            act = lambda x: x
        else:
            act = activation
        logits = fclayer(logits, layers_depth[i], layers_depth[i + 1], act, "fc_layer_" + str(i+1))

    tf.identity(logits, name='logits')

    with tf.name_scope("predictions"):
        predicted_label = tf.argmax(logits, 1, name='predicted_label')
        correct_prediction = tf.equal(predicted_label, y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        save_path = saver.save(sess,
                               modelvarnames['models_dir'] + \
                               modelvarnames['graph_directory'] + \
                               modelvarnames['graph_file'].split('.meta')[0])


class Model(object):

    def __init__(self, modelvarnames, restore=True):

        if modelvarnames['model2load']=='fcnncifar':
            fcnn([int(32*32*3), 1000, 250, 10], modelvarnames)
            self.nlayers = 3
        else:
            self.nlayers = 3

        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        self.graph = tf.get_default_graph()
        self.saver = tf.train.import_meta_graph(modelvarnames['models_dir'] + \
                                                modelvarnames['graph_directory'] + \
                                                modelvarnames['graph_file'])

        self.Xtrue = self.graph.get_tensor_by_name(modelvarnames['input'])
        self.Ylogits = self.graph.get_tensor_by_name(modelvarnames['logits'])
        if modelvarnames['pkeep'] == None:
            self.pkeep = tf.placeholder(tf.int32, (None))  # Not used

        else:
            self.pkeep = self.graph.get_tensor_by_name(modelvarnames['pkeep'])

        # self.Ytrue = tf.placeholder(tf.int32, (None))
        self.Ytrue = self.graph.get_tensor_by_name('y:0')

        idx = np.arange(0,10, dtype=np.int32)
        # self.margin =tf.gather(self.Ylogits, self.Ytrue) - tf.gather(self.Ylogits, tf.setdiff1d(idx,self.Ytrue))


        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.Ylogits,
                                                            labels=tf.one_hot(self.Ytrue, 10))
        self.grad_loss = tf.gradients(self.loss, self.Xtrue)[0]

        self.learning_rate = modelvarnames['lr']
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.loss_avg = tf.reduce_mean(self.loss)
        self.trainop = self.optimizer.minimize(self.loss_avg)

        self.acc = self.graph.get_tensor_by_name('predictions/accuracy:0')
        self.sess.run(tf.global_variables_initializer())

        if restore:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(modelvarnames['models_dir'] + \
                                                                     modelvarnames['graph_directory']))

class LayerNorms(object):
    def __init__(self, model, layer=1):
        Wtf = model.graph.get_tensor_by_name('fc_layer_'+str(layer)+'/W:0')
        W = model.sess.run(Wtf)
        # print(W.shape)

        sigma = np.linalg.svd(W,compute_uv=False)

        self.mixed_half_inf = (np.sum(np.sqrt(np.abs(W)), axis=1) ** 2.).max()
        self.mixed_1_inf = np.sum(np.abs(W), axis=1).max()
        self.mixed_1_1 = np.sum(np.abs(W), axis=1).sum()
        self.mixed_1_2 = np.sqrt( (np.sum(np.abs(W), axis=1)**2.).sum())
        self.mixed_2_1 = np.sqrt( np.sum(np.abs(W)**2., axis=1)).sum()
        self.fro = np.sqrt(np.sum(W.flatten()**2.))
        self.spectral = np.max(sigma)

def get_err(model, Xtest, ytest):
    acc = model.sess.run(model.acc, feed_dict={
        model.Xtrue: Xtest,
        model.Ytrue: ytest,
        model.pkeep: 1.0
    })
    err = (1. - acc)
    return err


def next_batch(batch_size, X=None, y=None):
    Xbatch, ybatch = shuffle(X, y)
    Xbatch = Xbatch[:batch_size]
    ybatch = ybatch[:batch_size]

    return Xbatch, ybatch

def main():
    args = get_arguments()

    modelvarnames = get_all_model_variables(args)
    # Load Dataset
    # X = mnist.train.images
    # y = mnist.train.labels
    # Xtest = mnist.test.images
    # ytest = mnist.test.labels
    # mnist = None # Free Memory
    if modelvarnames['model2load']=='fcnnmnist':
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets(modelvarnames['data_dir'], one_hot=True)
        Xtrain, ytrain = mnist.train.next_batch(modelvarnames['batch-size'])
        Xtest, ytest = mnist.test.next_batch(modelvarnames['n_images'])
    elif modelvarnames['model2load']=='fcnncifar':
        from cifar10 import maybe_download_and_extract, load_training_data, load_test_data
        maybe_download_and_extract()
        Xtrain, class_labels_train, ytrain = load_training_data()
        Xtest, class_labels_test, ytest = load_test_data()

    else:
        print('error loading datasets')
        return

    # pre-process data
    Xtrain, ytrain = pre_process_data(Xtrain, ytrain, modelvarnames['model2load'])
    Xtest, ytest = pre_process_data(Xtest, ytest, modelvarnames['model2load'])

    print(ytest)
    # Load pre-trained model
    model = Model(modelvarnames, restore=False)

    # Create necesary directories
    if not os.path.exists('./results/'):
        os.makedirs('./results/')


    # Set up simulation parameters
    eps_rescale = np.max(np.abs( np.max(Xtrain.flatten()) -  np.min(Xtrain.flatten()) ))
    epsilon = modelvarnames['min_epsilon'] * eps_rescale

    print('Computing fooling ratios...')
    print()

    iterations = int(5e4/modelvarnames['batch-size']*modelvarnames['epochs'])
    time = np.unique(np.linspace(0,iterations,100 +1).astype(int))
    t=0
    eps_test = modelvarnames['max_epsilon'] * eps_rescale
    FLAG_adv_train = False

    # Initialize Result matrices
    summary_dict = {'adv-training': 0,
                    'std-train-error': 1,
                    'adv-train-error': 2,
                    'std-test-error': 3,
                    'adv-test-error': 4
                    # 'std-min-margin': 5,
                    # 'adv-min-margin': 6,
                    # 'std-mean-margin': 7,
                    # 'adv-mean-margin': 8
                    }
    summary_mtx = np.zeros([len(time), len(summary_dict)])

    norm_mtx_list = []
    norm_dict = {
                'mixed-half-inf': 0,
                'mixed-1-inf': 1,
                'mixed-1-1': 2,
                'fro': 3,
                'spectral':4,
                'mixed-1-2':5,
                'mixed-2-1':6
                }
    for jj in range(model.nlayers):
        norm_mtx_list.append(np.zeros([len(time), len(norm_dict)]))


    # Start Simulation
    for ii in range(iterations):
        if modelvarnames['model2load'] == 'fcnnmnist':
            X, y = mnist.train.next_batch(modelvarnames['batch-size'])
            X, y = pre_process_data(X, y, modelvarnames['model2load'])
        else:
            X, y = next_batch(modelvarnames['batch-size'], Xtrain, ytrain)

        if FLAG_adv_train: # Add Adv noise only if AdvTraining has started

            X = get_adversarial_examples(X, y, model, epsilon, iterations=modelvarnames['its-advtrain'])
            acc_train = model.sess.run(model.acc, feed_dict={
                model.Xtrue: X,
                model.Ytrue: y,
                model.pkeep: 1.0
            })
            print('adv-acc = {}'.format(acc_train))
            if acc_train>=modelvarnames['adv-acc-thrs']:
                if epsilon >= modelvarnames['max_epsilon'] * eps_rescale: # max epsilon reached --> increase its
                    modelvarnames['its-advtrain'] = np.minimum(modelvarnames['its-advtrain']+1, 10)
                epsilon *= 1.1
                epsilon = np.minimum(epsilon, modelvarnames['max_epsilon'] * eps_rescale)
                print('\n\n eps = {}, its = {} \n\n'.format(epsilon, modelvarnames['its-advtrain']))

        model.sess.run(model.trainop, feed_dict={model.Xtrue: X, model.Ytrue:y, model.pkeep:1.0})

        # Run training operation (update the weights)
        if ii in time: # Get performance indicators and norms of weights

            for jj in range(model.nlayers): # Store norms of weights
                layer_norms = LayerNorms(model, layer=jj+1)
                norm_mtx_list[jj][t, norm_dict['mixed-half-inf']] = layer_norms.mixed_half_inf
                norm_mtx_list[jj][t, norm_dict['mixed-1-inf']] = layer_norms.mixed_1_inf
                norm_mtx_list[jj][t, norm_dict['mixed-1-1']] = layer_norms.mixed_1_1
                norm_mtx_list[jj][t, norm_dict['fro']] = layer_norms.fro
                norm_mtx_list[jj][t, norm_dict['spectral']] = layer_norms.spectral
                norm_mtx_list[jj][t, norm_dict['mixed-1-2']] = layer_norms.mixed_1_2
                norm_mtx_list[jj][t, norm_dict['mixed-2-1']] = layer_norms.mixed_2_1

            # Load a batch of images from the TEST set
            if modelvarnames['model2load'] == 'fcnnmnist':
                X, y = mnist.test.next_batch(modelvarnames['n_images'])
                X, y = pre_process_data(X, y, modelvarnames['model2load'])
            else:
                X, y = next_batch(modelvarnames['n_images'], Xtest, ytest)
            # Compute the TEST accuracy WITHOUT adversarial noise
            summary_mtx[t, summary_dict['std-test-error']] = get_err(model, X, y)


            # Compute the TEST accuracy WITH adversarial noise
            X = get_adversarial_examples(X, y, model, eps_test, iterations=10)
            summary_mtx[t, summary_dict['adv-test-error']] = get_err(model, X, y)


            # Load a batch of images from the TRAIN set
            if modelvarnames['model2load'] == 'fcnnmnist':
                X, y = mnist.train.next_batch(modelvarnames['n_images'])
                X, y = pre_process_data(X, y, modelvarnames['model2load'])
            else:
                X, y = next_batch(modelvarnames['n_images'], Xtrain, ytrain)

            # Compute the TRAIN accuracy WITHOUT adversarial noise
            summary_mtx[t, summary_dict['std-train-error']] = get_err(model, X, y)

            # Compute the TRAIN accuracy WITH adversarial noise
            X = get_adversarial_examples(X, y, model, eps_test, iterations=10)
            summary_mtx[t, summary_dict['adv-train-error']] = get_err(model, X, y)

            summary_mtx[t, summary_dict['adv-training']] = int(FLAG_adv_train)


            print('{:.0f}% done.. {}'.format(ii/iterations*100, summary_mtx[t,:]))

            if ii/iterations > modelvarnames['start-advtrain']:  # We start adv training at 50% of the iterations
                FLAG_adv_train = True

            # Load a batch of images from the TEST set
            t+=1

    # Save training/test errors
    ID = str(np.random.randint(0, int(1e5)))
    np.savetxt('./results/ID-{}_{}-summary.csv'.format(ID, modelvarnames['model2load']),
               summary_mtx, delimiter=';')

    # Save norms of weights
    for jj in range(model.nlayers):
        np.savetxt('./results/ID-{}_{}-W_{:.0f}.csv'.format(ID, modelvarnames['model2load'], jj+1),
                   norm_mtx_list[jj], delimiter=';')

    if not os.path.exists('./hyperparams/'):
        os.makedirs('./hyperparams/')

    hyperparams = np.array([modelvarnames['start-advtrain'],
                            modelvarnames['min_epsilon'],
                            modelvarnames['max_epsilon'],
                            modelvarnames['batch-size'],
                            modelvarnames['epochs'],
                            modelvarnames['adv-acc-thrs'],
                            modelvarnames['lr']
                            ]).astype(float)
    np.savetxt('./hyperparams/ID-{}_{}.csv'.format(ID, modelvarnames['model2load']),
               hyperparams, delimiter=';')

if __name__ == '__main__':
    main()