#!/usr/bin/python
import cPickle, getopt, sys, time, re
import datetime, os;

import scipy.io;
import nltk;
import numpy;
import matplotlib;
import matplotlib.pyplot;
import optparse;

"""
center the data, i.e., subtract the mean
"""
def center_data(data):
    (N, D) = data.shape;
    data = data - numpy.tile(data.mean(axis=0), (N, 1));
    return data

#training_iterations=100,
#input_directory="/Users/kzhai/Workspace/PyHogwarts/input/ap",
#output_directory="/Users/kzhai/Workspace/PyHogwarts/output",

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        
                        # parameter set 2
                        training_iterations=-1,
                        snapshot_interval=10,

                        # parameter set 3
                        alpha=1.,
                        sigma_a=1.,
                        sigma_x=1.,
                        
                        # parameter set 4
                        # disable_alpha_theta_update=False,
                        inference_mode=1,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    # parser.add_option("--dictionary", type="string", dest="dictionary",
                      # help="the dictionary file [None]")
    
    # parameter set 2
    # parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
                      # help="total number of topics [-1]");
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="total number of iterations [-1]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [10]");                      
                      
    # parameter set 3
    parser.add_option("--alpha", type="float", dest="alpha",
                      help="hyper-parameter for alpha [1.0]")
    parser.add_option("--sigma_a", type="float", dest="sigma_a",
                      help="hyper-parameter for Sigma A [1.0]")
    parser.add_option("--sigma_x", type="float", dest="sigma_x",
                      help="hyper-parameter for Sigma X [1.0]")
    
    # parameter set 4
    # parser.add_option("--disable_alpha_theta_update", action="store_true", dest="disable_alpha_theta_update",
                      # help="disable alpha (hyper-parameter for Dirichlet distribution of topics) update");
    parser.add_option("--inference_mode", type="int", dest="inference_mode",
                      help="inference mode [ " + 
                            "1 (default): monte carlo, " + 
                            "2: variational bayes " + 
                            "]");
    # parser.add_option("--inference_mode", action="store_true", dest="inference_mode",
    #                  help="run latent Dirichlet allocation in lda mode");

    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();

    # parameter set 2
    # assert(options.number_of_topics>0);
    # number_of_topics = options.number_of_topics;
    assert(options.training_iterations > 0);
    training_iterations = options.training_iterations;
    assert(options.snapshot_interval > 0);
    if options.snapshot_interval > 0:
        snapshot_interval = options.snapshot_interval;
    
    # parameter set 4
    # disable_alpha_theta_update = options.disable_alpha_theta_update;
    inference_mode = options.inference_mode;
    
    # parameter set 1
    # assert(options.dataset_name!=None);
    assert(options.input_directory != None);
    assert(options.output_directory != None);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    dataset_name = os.path.basename(input_directory);
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, dataset_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);

    # Dataset
    train_file_path = os.path.join(input_directory, 'train.dat')
    train_data = numpy.loadtxt(train_file_path)
    train_data = center_data(train_data);
    print "successfully load all train_data from %s..." % (os.path.abspath(train_file_path));
    
    # parameter set 3
    assert(options.alpha > 0);
    alpha = options.alpha;
    assert(options.sigma_a > 0);
    sigma_a = options.sigma_a;
    assert(options.sigma_x > 0);
    sigma_x = options.sigma_x;

    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("ibp");
    suffix += "-I%d" % (training_iterations);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-a%f" % (alpha);
    suffix += "-sa%f" % (sigma_a);
    suffix += "-sx%f" % (sigma_x);
    suffix += "-im%d" % (inference_mode);
    # suffix += "-%s" % (resample_topics);
    # suffix += "-%s" % (hash_oov_words);
    suffix += "/";
    
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));

    # dict_file = options.dictionary;
    # if dict_file != None:
        # dict_file = dict_file.strip();
        
    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("dataset_name=" + dataset_name + "\n");
    # options_output_file.write("vocabulary_path=" + str(dict_file) + "\n");
    # parameter set 2
    options_output_file.write("training_iterations=%d\n" % (training_iterations));
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");
    # options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("alpha=" + str(alpha) + "\n");
    options_output_file.write("sigma_a=" + str(sigma_a) + "\n");
    options_output_file.write("sigma_x=" + str(sigma_x) + "\n");
    # parameter set 4
    options_output_file.write("inference_mode=%d\n" % (inference_mode));
    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "dataset_name=" + dataset_name
    # print "dictionary file=" + str(dict_file)
    # parameter set 2
    print "training_iterations=%d" % (training_iterations);
    print "snapshot_interval=" + str(snapshot_interval);
    # print "number_of_topics=" + str(number_of_topics)
    # parameter set 3
    print "alpha=" + str(alpha)
    print "sigma_a=" + str(sigma_a)
    print "sigma_x=" + str(sigma_x)
    # parameter set 4
    print "inference_mode=%d" % (inference_mode)
    print "========== ========== ========== ========== =========="
    
    # if inference_mode==0:
        # import hybrid
        # ibp_inferencer = hybrid.Hybrid();
    if inference_mode == 1:
        import collapsed_gibbs
        
        # initialize the model
        ibp_inferencer = collapsed_gibbs.CollapsedGibbs();
        # ibp_inferencer = monte_carlo.MonteCarlo(alpha_hyper_parameter, sigma_x_hyper_parameter, sigma_a_hyper_parameter, True);
    elif inference_mode == 2:
        import uncollapsed_gibbs
        
        # initialize the model
        ibp_inferencer = uncollapsed_gibbs.UncollapsedGibbs();
        # ibp_inferencer = monte_carlo.MonteCarlo(alpha_hyper_parameter, sigma_x_hyper_parameter, sigma_a_hyper_parameter, True);
    elif inference_mode == 3:
        import semicollapsed_gibbs
        
        # initialize the model
        ibp_inferencer = semicollapsed_gibbs.SemiCollapsedGibbs();
        # ibp_inferencer = monte_carlo.MonteCarlo(alpha_hyper_parameter, sigma_x_hyper_parameter, sigma_a_hyper_parameter, True);
    elif inference_mode == 3:
        import variational_bayes
        ibp_inferencer = variational_bayes.VariationalBayes();
    else:
        sys.stderr.write("error: unrecognized inference mode %d...\n" % (inference_mode));
        return;
    
    ibp_inferencer._initialize(train_data[:100, :], sigma_a, sigma_x, initial_Z=None, A_prior=None);
    
    for iteration in xrange(training_iterations):
        log_likelihood = ibp_inferencer.learning();
    
        print "iteration: %i\tK: %i\tlikelihood: %f" % (ibp_inferencer._counter, ibp_inferencer._K, log_likelihood);
        
        if (ibp_inferencer._counter % snapshot_interval == 0):
            ibp_inferencer.export_snapshot(output_directory);

        print ibp_inferencer._Z.sum(axis=0)
        
    model_snapshot_path = os.path.join(output_directory, 'model-' + str(ibp_inferencer._counter));
    cPickle.dump(ibp_inferencer, open(model_snapshot_path, 'wb'));

    '''
    # If matplotlib is installed, plot ground truth vs learned factors
    import matplotlib.pyplot as P
    from scaled_image import scaledimage
    
    # Intensity plots of
    # -ground truth factor-feature weights (top)
    # -learned factor-feature weights (bottom)
    K = max(len(true_weights), len(ibp._A))
    (fig, subaxes) = matplotlib.pyplot.subplots(2, K)
    for sa in subaxes.flatten():
        sa.set_visible(False)
    fig.suptitle('Ground truth (top) vs learned factors (bottom)')
    for (idx, trueFactor) in enumerate(true_weights):
        ax = subaxes[0, idx]
        ax.set_visible(True)
        scaledimage(trueFactor.reshape(6, 6),
                    pixwidth=3, ax=ax)
    for (idx, learnedFactor) in enumerate(ibp._A):
        ax = subaxes[1, idx]
        scaledimage(learnedFactor.reshape(6, 6),
                    pixwidth=3, ax=ax)
        ax.set_visible(True)
    matplotlib.pyplot.show()
    '''
    
"""
run IBP on the synthetic 'cambridge bars' dataset, used in the original paper.
"""
if __name__ == '__main__':
    main()
    
    '''
    # load the data from the matrix
    mat_vals = scipy.io.loadmat('./cambridge-bars/block_image_set.mat');
    true_weights = mat_vals['trueWeights']
    features = mat_vals['features']
    data = mat_vals['data']
    
    # set up the hyper-parameter for sampling alpha
    alpha_hyper_parameter = (1., 1.);
    # set up the hyper-parameter for sampling sigma_x
    sigma_x_hyper_parameter = (1., 1.);
    # set up the hyper-parameter for sampling sigma_a
    sigma_a_hyper_parameter = (1., 1.);
    
    features = features.astype(numpy.int);
    
    # initialize the model
    ibp = CollapsedGibbsSampling(alpha_hyper_parameter, sigma_x_hyper_parameter, sigma_a_hyper_parameter, True);

    ibp._initialize(data[1:100, :], 0.5, 0.2, 0.5);
    #ibp._initialize(data[0:1000, :], 1.0, 1.0, 1.0, None, features[0:1000, :]);
    
    #print ibp._Z, "\n", ibp._A_mean
    ibp.sample(30);
    
    print ibp._Z.sum(axis=0)

    # If matplotlib is installed, plot ground truth vs learned factors
    import matplotlib.pyplot as P
    from scaled_image import scaledimage
    
    # Intensity plots of
    # -ground truth factor-feature weights (top)
    # -learned factor-feature weights (bottom)
    K = max(len(true_weights), len(ibp._A))
    (fig, subaxes) = matplotlib.pyplot.subplots(2, K)
    for sa in subaxes.flatten():
        sa.set_visible(False)
    fig.suptitle('Ground truth (top) vs learned factors (bottom)')
    for (idx, trueFactor) in enumerate(true_weights):
        ax = subaxes[0, idx]
        ax.set_visible(True)
        scaledimage(trueFactor.reshape(6, 6),
                    pixwidth=3, ax=ax)
    for (idx, learnedFactor) in enumerate(ibp._A):
        ax = subaxes[1, idx]
        scaledimage(learnedFactor.reshape(6, 6),
                    pixwidth=3, ax=ax)
        ax.set_visible(True)
    matplotlib.pyplot.show()
    '''