import sys
import numpy
import matplotlib.pyplot
import matplotlib.ticker
import matplotlib.cm
import cPickle
import optparse
import os

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        #input_directory=None,
                        model_directory=None,
                        snapshot_index=-1,
                        )
    # parameter set 1
    #parser.add_option("--input_directory", type="string", dest="input_directory",
                      #help="input directory [None]");
    parser.add_option("--model_directory", type="string", dest="model_directory",
                      help="model directory [None]");
    parser.add_option("--snapshot_index", type="int", dest="snapshot_index",
                      help="snapshot index [-: evaluate on all available snapshots]");
    
    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();
    
    # parameter set 1
    assert(options.model_directory!=None);
    
    model_directory = options.model_directory;
    if not os.path.exists(model_directory):
        sys.stderr.write("error: model directory %s does not exist...\n" % (os.path.abspath(model_directory)));
        return;

    snapshot_index=options.snapshot_index;

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "model_directory=" + model_directory
    print "snapshot_index=" + str(snapshot_index);
    print "========== ========== ========== ========== =========="
    
    for file_name in os.listdir(model_directory):
        if not file_name.startswith("model"):
            continue;
        
        input_snapshot_path = os.path.join(model_directory, file_name);
        print input_snapshot_path
        
        ibp_inferencer = cPickle.load(open(input_snapshot_path, "rb" ));
        
        # Intensity plots of
        # -ground truth factor-feature weights (top)
        # -learned factor-feature weights (bottom)
        #K = max(len(true_weights), len(ibp_inferencer._A))

        '''        
        fig, ax = matplotlib.pyplot.subplots()

        image = numpy.reshape(ibp_inferencer._A, (6, 6));
        ax.imshow(image, cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
        ax.set_title('dropped spines')
        
        # Move left and bottom spines outward by 10 points
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        '''

        matrix_A = ibp_inferencer._A;
        #matrix_A = ibp_inferencer._phi_mean;
        
        K = len(matrix_A);
        (fig, subaxes) = matplotlib.pyplot.subplots(1, K)
        for sa in subaxes.flatten():
            sa.set_visible(False)
            
        for (idx, learnedFactor) in enumerate(matrix_A):
            print idx, learnedFactor
            ax = subaxes[idx]
            scaled_image(learnedFactor.reshape(6, 6), pixwidth=3, ax=ax)
            ax.set_visible(True)

        '''
        fig.suptitle('Ground truth (top) vs learned factors (bottom)')
        for (idx, trueFactor) in enumerate(true_weights):
            ax = subaxes[1, idx]
            ax.set_visible(True)
            scaled_image(trueFactor.reshape(6, 6), pixwidth=3, ax=ax)
        '''
            
        matplotlib.pyplot.show()
  
def scaled_image(W, pixwidth=1, ax=None, grayscale=True):
    """
    Do intensity plot, similar to MATLAB imagesc()
    
    W = intensity matrix to visualize
    pixwidth = size of each W element
    ax = matplotlib Axes to draw on
    grayscale = use grayscale color map
    
    Rely on caller to .show()
    """
    
    # N = rows, M = column
    (N, M) = W.shape
    # Need to create a new Axes?
    if(ax == None):
        ax = matplotlib.pyplot.figure().gca()
    # extents = Left Right Bottom Top
    exts = (0, pixwidth * M, 0, pixwidth * N)
    if(grayscale):
        ax.imshow(W,
                  interpolation='nearest',
                  cmap=matplotlib.cm.gray,
                  extent=exts)
    else:
        ax.imshow(W,
                  interpolation='nearest',
                  extent=exts)

    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    return ax

if __name__ == '__main__':
    main();