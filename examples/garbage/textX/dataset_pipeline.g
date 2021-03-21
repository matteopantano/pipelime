graph {

  _:A -> multiplexer:_input_dataset[0];
  _:B -> multiplexer:_input_dataset[1];
  _:C -> multiplexer:_input_dataset[2];

  _:xray_params -> xray0:params;
  _:xray_params -> xray1:params;
  _:xray_params -> xray2:params;

  multiplexer:_output_dataset[0] -> xray0:_input_dataset;
  multiplexer:_output_dataset[1] -> xray1:_input_dataset;
  multiplexer:_output_dataset[2] -> xray2:_input_dataset;
  

  xray0:_output_dataset -> check0:_input_dataset;
  xray1:_output_dataset -> check1:_input_dataset;
  xray2:_output_dataset -> check2:_input_dataset;
  
  check0:_output_dataset -> merge0:_in[0];
  check1:_output_dataset -> merge0:_in[1];
  check2:_output_dataset -> merge0:_in[2];

  _:A -> merge1:_in[0];
  _:B -> merge1:_in[1];
  _:C -> merge1:_in[2];

  merge0:_output_dataset -> merge2:_in[0];
  merge1:_output_dataset -> merge2:_in[1];

  merge2:_output_dataset -> split:_in;

  split:_output_dataset['train'] -> O:d_train;
  split:_output_dataset['test'] -> O:d_test;
  split:_output_dataset['val'] -> O:d_val;
}