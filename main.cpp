/*
 * File:   main.cpp
 * Author: Bhupendra Raut
 *
 * Created on March 14, 2013, 3:07 PM
 */

#include <cstdlib>
#include <iostream>
//Fast ANN libs c++ interface
//#include <doublefann.h>    //if you are using 'double' data type
#include <floatfann.h>         //if you are using float only (for this application)
#include <fann_cpp.h>
#include <ios>
#include <iostream>
#include <iomanip>

using namespace std;

/*
 * ANN training design to take mean area precipitation for the domain and SLP data as input and AWAP clusters as output.
 * The Network had a biase neuron to avoid overtraining/saturation
 * The Transfer function use here is Sigmoid Asysmetric [0,1]. It is good to avoid 'zero' as input value (Just a fear!).
 */
int
main (int argc, char** argv)
{
  time_t start_t = time (NULL);
  const float learning_rate = 0.7f;
  const unsigned int num_layers = 3; //including input and output layers
  const unsigned int num_input = 4; //number of input to the ann
  const unsigned int num_neurons_hidden1 = 20; //neurons in hidden layer1 between input  and output
  //const unsigned int num_neurons_hidden2 = 5; //neurons in hidden layer2 between input  and output
  const unsigned int num_output = 5; //number of output to the ann
  const float desired_error = 0.01f;
  const unsigned int max_epochs = 10000; // iterations for training
  const unsigned int epochs_between_reports = 1000; //will print relevant information  every 10000 iterations
  struct fann *myNetwork;                               //c struct (It is annoying to work with c struct but c++ interface is crap)
  struct fann_train_data *data;                         //data struct (should find out how to change this for custom input method)
  //unsigned int i = 0;
  unsigned int decimal_point;
  fann_type *calc_out;

  cout << endl << "Creating network ...";
  myNetwork = fann_create_standard (num_layers, num_input, num_neurons_hidden1, num_output);
  cout << "\tSuccess" << endl;
  cout<< "getting training data ...";
  data = fann_read_train_from_file ("/Users/bhupendra/Dropbox/for_training/training_dataset_MAP_SLP.txt"); //("/Users/bhupendra/Dropbox/for_training/training_dataset_MAP_SLP.txt");
  cout << "\tSuccess" << endl;
  cout<< "setting ANN properties ...";
  fann_set_activation_steepness_hidden (myNetwork, 0.5);
  fann_set_activation_steepness_output (myNetwork, 0.5);
  fann_set_activation_function_hidden (myNetwork, FANN_SIGMOID);
  fann_set_activation_function_output (myNetwork, FANN_SIGMOID);
  //fann_set_train_stop_function (myNetwork, FANN_STOPFUNC_BIT);
  //fann_set_bit_fail_limit (myNetwork, 0.01f);
  //fann_set_training_algorithm (myNetwork, FANN_TRAIN_RPROP);
  cout << "\tSuccess" << endl;
  cout << "initializing weights ...";
  fann_init_weights (myNetwork, data);
  cout << "\tSuccess" << endl;

  printf ("Training network.\n");
  fann_train_on_data (myNetwork, data, max_epochs, epochs_between_reports, desired_error);
  printf ("Testing network. %f\n", fann_test_data (myNetwork, data));

  for (int i = 0; i < fann_length_train_data (data); i++)
    {
      calc_out = fann_run (myNetwork, data->input[i]);
      printf ("test (%4.2f) -> %4.2f %4.2f %4.2f %4.2f %4.2f, should be %4.2f %4.2f %4.2f %4.2f %4.2f,\n",
              data->input[i][0],
              calc_out[0], calc_out[1], calc_out[2], calc_out[3], calc_out[4],
              data->output[i][0], data->output[i][1], data->output[i][2], data->output[i][3], data->output[i][4]);
    }

  printf ("Saving network.\n");
  fann_save (myNetwork, "trained.net");

  decimal_point = fann_save_to_fixed (myNetwork, "/Users/bhupendra/Dropbox/for_training/training_dataset_MAP_SLP_NETWORK.txt");
  fann_save_train_to_fixed (data, "/Users/bhupendra/Dropbox/for_training/training_dataset_MAP_SLP_trained_fixed.data", decimal_point);

  printf ("Cleaning up.\n");
  fann_destroy_train (data);
  fann_destroy (myNetwork);

  time_t end_t = time (NULL);
  int runTime = end_t - start_t; //get_niceTime (end_t - start_t);
  cout << "Total time for this run was " << runTime << " seconds."<<endl;
}