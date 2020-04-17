TensorFlow 2.0 Upgrade Script
-----------------------------
Converted 14 files
Detected 30 issues that require attention
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
File: convLSTM_predicttimeseries/sofilstm/convlstm.py
--------------------------------------------------------------------------------
convLSTM_predicttimeseries/sofilstm/convlstm.py:58:15: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
convLSTM_predicttimeseries/sofilstm/convlstm.py:112:8: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
convLSTM_predicttimeseries/sofilstm/convlstm.py:115:11: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
convLSTM_predicttimeseries/sofilstm/convlstm.py:119:11: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
convLSTM_predicttimeseries/sofilstm/convlstm.py:120:11: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
convLSTM_predicttimeseries/sofilstm/convlstm.py:123:10: WARNING: tf.contrib.layers.layer_norm requires manual check. (Manual edit required) `tf.contrib.layers.layer_norm` has been deprecated, and its implementation has been integrated with `tf.keras.layers.LayerNormalization` in TensorFlow 2.0. Note that, the default value of `epsilon` is changed to `1e-3` in the new API from `1e-12`, and this may introduce numerical differences. Please check the new API and use that instead.
convLSTM_predicttimeseries/sofilstm/convlstm.py:123:10: ERROR: Using member tf.contrib.layers.layer_norm in deprecated module tf.contrib. tf.contrib.layers.layer_norm cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
convLSTM_predicttimeseries/sofilstm/convlstm.py:124:10: WARNING: tf.contrib.layers.layer_norm requires manual check. (Manual edit required) `tf.contrib.layers.layer_norm` has been deprecated, and its implementation has been integrated with `tf.keras.layers.LayerNormalization` in TensorFlow 2.0. Note that, the default value of `epsilon` is changed to `1e-3` in the new API from `1e-12`, and this may introduce numerical differences. Please check the new API and use that instead.
convLSTM_predicttimeseries/sofilstm/convlstm.py:124:10: ERROR: Using member tf.contrib.layers.layer_norm in deprecated module tf.contrib. tf.contrib.layers.layer_norm cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
convLSTM_predicttimeseries/sofilstm/convlstm.py:125:10: WARNING: tf.contrib.layers.layer_norm requires manual check. (Manual edit required) `tf.contrib.layers.layer_norm` has been deprecated, and its implementation has been integrated with `tf.keras.layers.LayerNormalization` in TensorFlow 2.0. Note that, the default value of `epsilon` is changed to `1e-3` in the new API from `1e-12`, and this may introduce numerical differences. Please check the new API and use that instead.
convLSTM_predicttimeseries/sofilstm/convlstm.py:125:10: ERROR: Using member tf.contrib.layers.layer_norm in deprecated module tf.contrib. tf.contrib.layers.layer_norm cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
convLSTM_predicttimeseries/sofilstm/convlstm.py:132:11: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
convLSTM_predicttimeseries/sofilstm/convlstm.py:135:10: WARNING: tf.contrib.layers.layer_norm requires manual check. (Manual edit required) `tf.contrib.layers.layer_norm` has been deprecated, and its implementation has been integrated with `tf.keras.layers.LayerNormalization` in TensorFlow 2.0. Note that, the default value of `epsilon` is changed to `1e-3` in the new API from `1e-12`, and this may introduce numerical differences. Please check the new API and use that instead.
convLSTM_predicttimeseries/sofilstm/convlstm.py:135:10: ERROR: Using member tf.contrib.layers.layer_norm in deprecated module tf.contrib. tf.contrib.layers.layer_norm cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
convLSTM_predicttimeseries/sofilstm/convlstm.py:136:10: WARNING: tf.contrib.layers.layer_norm requires manual check. (Manual edit required) `tf.contrib.layers.layer_norm` has been deprecated, and its implementation has been integrated with `tf.keras.layers.LayerNormalization` in TensorFlow 2.0. Note that, the default value of `epsilon` is changed to `1e-3` in the new API from `1e-12`, and this may introduce numerical differences. Please check the new API and use that instead.
convLSTM_predicttimeseries/sofilstm/convlstm.py:136:10: ERROR: Using member tf.contrib.layers.layer_norm in deprecated module tf.contrib. tf.contrib.layers.layer_norm cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
--------------------------------------------------------------------------------
File: convLSTM_predicttimeseries/sofilstm/layers.py
--------------------------------------------------------------------------------
convLSTM_predicttimeseries/sofilstm/layers.py:52:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
convLSTM_predicttimeseries/sofilstm/layers.py:61:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
convLSTM_predicttimeseries/sofilstm/layers.py:70:18: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
convLSTM_predicttimeseries/sofilstm/layers.py:71:15: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
--------------------------------------------------------------------------------
File: convLSTM_predicttimeseries/sofilstm/layers_old.py
--------------------------------------------------------------------------------
convLSTM_predicttimeseries/sofilstm/layers_old.py:51:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
convLSTM_predicttimeseries/sofilstm/layers_old.py:77:14: ERROR: Using member tf.contrib.layers.conv2d_transpose in deprecated module tf.contrib. tf.contrib.layers.conv2d_transpose cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
convLSTM_predicttimeseries/sofilstm/layers_old.py:87:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
convLSTM_predicttimeseries/sofilstm/layers_old.py:95:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
convLSTM_predicttimeseries/sofilstm/layers_old.py:102:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
convLSTM_predicttimeseries/sofilstm/layers_old.py:127:20: WARNING: Using member tf.contrib.rnn.ConvLSTMCell in deprecated module tf.contrib.rnn. (Manual edit required) tf.contrib.rnn.* has been deprecated, and widely used cells/functions will be moved to tensorflow/addons repository. Please check it there and file Github issues if necessary.
convLSTM_predicttimeseries/sofilstm/layers_old.py:127:20: ERROR: Using member tf.contrib.rnn.ConvLSTMCell in deprecated module tf.contrib. tf.contrib.rnn.ConvLSTMCell cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
--------------------------------------------------------------------------------
File: convLSTM_predicttimeseries/sofilstm/sofi.py
--------------------------------------------------------------------------------
convLSTM_predicttimeseries/sofilstm/sofi.py:154:20: WARNING: *.save requires manual check. (This warning is only applicable if the code saves a tf.Keras model) Keras model.save now saves to the Tensorflow SavedModel format by default, instead of HDF5. To continue saving to HDF5, add the argument save_format='h5' to the save() function.
--------------------------------------------------------------------------------
File: convLSTM_predicttimeseries/sofilstm/train.py
--------------------------------------------------------------------------------
convLSTM_predicttimeseries/sofilstm/train.py:207:20: WARNING: *.save requires manual check. (This warning is only applicable if the code saves a tf.Keras model) Keras model.save now saves to the Tensorflow SavedModel format by default, instead of HDF5. To continue saving to HDF5, add the argument save_format='h5' to the save() function.
convLSTM_predicttimeseries/sofilstm/train.py:209:28: WARNING: *.save requires manual check. (This warning is only applicable if the code saves a tf.Keras model) Keras model.save now saves to the Tensorflow SavedModel format by default, instead of HDF5. To continue saving to HDF5, add the argument save_format='h5' to the save() function.
================================================================================
Detailed log follows:

================================================================================
================================================================================
Input tree: 'convLSTM_predicttimeseries/'
================================================================================
--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/mytrain_64_t50.py'
 outputting to 'convLSTM_predicttimeseries_v2/mytrain_64_t50.py'
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/prep_model.py'
 outputting to 'convLSTM_predicttimeseries_v2/prep_model.py'
--------------------------------------------------------------------------------

30:5: INFO: Renamed 'tf.Session' to 'tf.compat.v1.Session'
32:12: INFO: Renamed 'tf.train.import_meta_graph' to 'tf.compat.v1.train.import_meta_graph'
38:23: INFO: Renamed 'tf.graph_util.convert_variables_to_constants' to 'tf.compat.v1.graph_util.convert_variables_to_constants'
40:8: INFO: Renamed 'tf.get_default_graph' to 'tf.compat.v1.get_default_graph'
45:9: INFO: Renamed 'tf.gfile.GFile' to 'tf.io.gfile.GFile'
56:13: INFO: Renamed 'tf.GraphDef' to 'tf.compat.v1.GraphDef'
57:5: INFO: Renamed 'tf.gfile.Open' to 'tf.io.gfile.GFile'
69:4: INFO: Renamed 'tf.gfile.FastGFile' to 'tf.compat.v1.gfile.FastGFile'
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/test.py'
 outputting to 'convLSTM_predicttimeseries_v2/test.py'
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/sofilstm/__init__.py'
 outputting to 'convLSTM_predicttimeseries_v2/sofilstm/__init__.py'
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/sofilstm/convlstm.py'
 outputting to 'convLSTM_predicttimeseries_v2/sofilstm/convlstm.py'
--------------------------------------------------------------------------------

24:24: INFO: Renamed 'tf.contrib.rnn.RNNCell' to 'tf.compat.v1.nn.rnn_cell.RNNCell'
39:12: INFO: Renamed 'tf.contrib.rnn.LSTMStateTuple' to 'tf.nn.rnn_cell.LSTMStateTuple'
48:9: INFO: Renamed 'tf.variable_scope' to 'tf.compat.v1.variable_scope'
58:15: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
58:15: INFO: Renamed 'tf.get_variable' to 'tf.compat.v1.get_variable'
59:15: INFO: Added keywords to args of function 'tf.nn.conv2d'
59:15: INFO: Renamed keyword argument for tf.nn.conv2d from filter to filters
67:20: INFO: Renamed 'tf.contrib.rnn.LSTMStateTuple' to 'tf.nn.rnn_cell.LSTMStateTuple'
73:19: INFO: Renamed 'tf.nn.rnn_cell.RNNCell' to 'tf.compat.v1.nn.rnn_cell.RNNCell'
100:11: INFO: Renamed 'tf.nn.rnn_cell.LSTMStateTuple' to 'tf.compat.v1.nn.rnn_cell.LSTMStateTuple'
112:8: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
112:8: INFO: Renamed 'tf.get_variable' to 'tf.compat.v1.get_variable'
113:8: INFO: Added keywords to args of function 'tf.nn.convolution'
113:8: INFO: Renamed keyword argument for tf.nn.convolution from filter to filters
115:11: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
115:11: INFO: Renamed 'tf.get_variable' to 'tf.compat.v1.get_variable'
115:52: INFO: tf.zeros_initializer requires manual check. Initializers no longer have the dtype argument in the constructor or partition_info argument in the __call__ method.
The calls have been converted to compat.v1 for safety (even though they may already have been correct).
115:52: INFO: Renamed 'tf.zeros_initializer' to 'tf.compat.v1.zeros_initializer'
119:11: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
119:11: INFO: Renamed 'tf.get_variable' to 'tf.compat.v1.get_variable'
120:11: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
120:11: INFO: Renamed 'tf.get_variable' to 'tf.compat.v1.get_variable'
123:10: WARNING: tf.contrib.layers.layer_norm requires manual check. (Manual edit required) `tf.contrib.layers.layer_norm` has been deprecated, and its implementation has been integrated with `tf.keras.layers.LayerNormalization` in TensorFlow 2.0. Note that, the default value of `epsilon` is changed to `1e-3` in the new API from `1e-12`, and this may introduce numerical differences. Please check the new API and use that instead.
123:10: ERROR: Using member tf.contrib.layers.layer_norm in deprecated module tf.contrib. tf.contrib.layers.layer_norm cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
124:10: WARNING: tf.contrib.layers.layer_norm requires manual check. (Manual edit required) `tf.contrib.layers.layer_norm` has been deprecated, and its implementation has been integrated with `tf.keras.layers.LayerNormalization` in TensorFlow 2.0. Note that, the default value of `epsilon` is changed to `1e-3` in the new API from `1e-12`, and this may introduce numerical differences. Please check the new API and use that instead.
124:10: ERROR: Using member tf.contrib.layers.layer_norm in deprecated module tf.contrib. tf.contrib.layers.layer_norm cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
125:10: WARNING: tf.contrib.layers.layer_norm requires manual check. (Manual edit required) `tf.contrib.layers.layer_norm` has been deprecated, and its implementation has been integrated with `tf.keras.layers.LayerNormalization` in TensorFlow 2.0. Note that, the default value of `epsilon` is changed to `1e-3` in the new API from `1e-12`, and this may introduce numerical differences. Please check the new API and use that instead.
125:10: ERROR: Using member tf.contrib.layers.layer_norm in deprecated module tf.contrib. tf.contrib.layers.layer_norm cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
132:11: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
132:11: INFO: Renamed 'tf.get_variable' to 'tf.compat.v1.get_variable'
135:10: WARNING: tf.contrib.layers.layer_norm requires manual check. (Manual edit required) `tf.contrib.layers.layer_norm` has been deprecated, and its implementation has been integrated with `tf.keras.layers.LayerNormalization` in TensorFlow 2.0. Note that, the default value of `epsilon` is changed to `1e-3` in the new API from `1e-12`, and this may introduce numerical differences. Please check the new API and use that instead.
135:10: ERROR: Using member tf.contrib.layers.layer_norm in deprecated module tf.contrib. tf.contrib.layers.layer_norm cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
136:10: WARNING: tf.contrib.layers.layer_norm requires manual check. (Manual edit required) `tf.contrib.layers.layer_norm` has been deprecated, and its implementation has been integrated with `tf.keras.layers.LayerNormalization` in TensorFlow 2.0. Note that, the default value of `epsilon` is changed to `1e-3` in the new API from `1e-12`, and this may introduce numerical differences. Please check the new API and use that instead.
136:10: ERROR: Using member tf.contrib.layers.layer_norm in deprecated module tf.contrib. tf.contrib.layers.layer_norm cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
141:12: INFO: Renamed 'tf.nn.rnn_cell.LSTMStateTuple' to 'tf.compat.v1.nn.rnn_cell.LSTMStateTuple'
153:15: INFO: Renamed 'tf.placeholder' to 'tf.compat.v1.placeholder'
161:25: INFO: Renamed 'tf.nn.dynamic_rnn' to 'tf.compat.v1.nn.dynamic_rnn'
162:13: INFO: Renamed 'tf.Session' to 'tf.compat.v1.Session'
164:21: INFO: Renamed 'tf.global_variables_initializer' to 'tf.compat.v1.global_variables_initializer'
177:17: INFO: Renamed 'tf.placeholder' to 'tf.compat.v1.placeholder'
180:25: INFO: Renamed 'tf.nn.dynamic_rnn' to 'tf.compat.v1.nn.dynamic_rnn'
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/sofilstm/image_util.py'
 outputting to 'convLSTM_predicttimeseries_v2/sofilstm/image_util.py'
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/sofilstm/layers.py'
 outputting to 'convLSTM_predicttimeseries_v2/sofilstm/layers.py'
--------------------------------------------------------------------------------

26:14: INFO: Renamed 'tf.log' to 'tf.math.log'
27:16: INFO: Renamed 'tf.log' to 'tf.math.log'
31:14: INFO: Renamed 'tf.truncated_normal' to 'tf.random.truncated_normal'
35:11: INFO: Added keywords to args of function 'tf.reduce_max'
35:11: INFO: Renamed keyword argument for tf.reduce_max from keep_dims to keepdims
36:11: INFO: Added keywords to args of function 'tf.reduce_min'
36:11: INFO: Renamed keyword argument for tf.reduce_min from keep_dims to keepdims
46:9: INFO: Added keywords to args of function 'tf.reduce_sum'
46:23: INFO: Renamed 'tf.squared_difference' to 'tf.math.squared_difference'
47:15: INFO: Changed tf.to_float call to tf.cast(..., dtype=tf.float32).
52:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
54:54: INFO: Renamed 'tf.layers.batch_normalization' to 'tf.compat.v1.layers.batch_normalization'
58:11: INFO: Changing keep_prob arg of tf.nn.dropout to rate, and recomputing value.

61:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
70:18: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
70:18: INFO: Renamed 'tf.get_variable' to 'tf.compat.v1.get_variable'
70:109: INFO: tf.initializers.glorot_uniform requires manual check. Initializers no longer have the dtype argument in the constructor or partition_info argument in the __call__ method.
The calls have been converted to compat.v1 for safety (even though they may already have been correct).
70:109: INFO: Renamed 'tf.initializers.glorot_uniform' to 'tf.compat.v1.initializers.glorot_uniform'
71:15: WARNING: tf.get_variable requires manual check. tf.get_variable returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().
71:15: INFO: Renamed 'tf.get_variable' to 'tf.compat.v1.get_variable'
71:82: INFO: tf.initializers.glorot_uniform requires manual check. Initializers no longer have the dtype argument in the constructor or partition_info argument in the __call__ method.
The calls have been converted to compat.v1 for safety (even though they may already have been correct).
71:82: INFO: Renamed 'tf.initializers.glorot_uniform' to 'tf.compat.v1.initializers.glorot_uniform'
72:10: INFO: Added keywords to args of function 'tf.nn.conv2d'
72:10: INFO: Renamed keyword argument for tf.nn.conv2d from filter to filters
74:10: INFO: Renamed 'tf.layers.batch_normalization' to 'tf.compat.v1.layers.batch_normalization'
76:11: INFO: Changing keep_prob arg of tf.nn.dropout to rate, and recomputing value.

97:11: INFO: Added keywords to args of function 'tf.nn.max_pool'
97:11: INFO: Renamed keyword argument for tf.nn.max_pool from value to input
97:11: INFO: Renamed 'tf.nn.max_pool' to 'tf.nn.max_pool2d'
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/sofilstm/layers_old.py'
 outputting to 'convLSTM_predicttimeseries_v2/sofilstm/layers_old.py'
--------------------------------------------------------------------------------

26:14: INFO: Renamed 'tf.log' to 'tf.math.log'
27:16: INFO: Renamed 'tf.log' to 'tf.math.log'
31:14: INFO: Renamed 'tf.truncated_normal' to 'tf.random.truncated_normal'
35:11: INFO: Added keywords to args of function 'tf.reduce_max'
35:11: INFO: Renamed keyword argument for tf.reduce_max from keep_dims to keepdims
36:11: INFO: Added keywords to args of function 'tf.reduce_min'
36:11: INFO: Renamed keyword argument for tf.reduce_min from keep_dims to keepdims
46:9: INFO: Added keywords to args of function 'tf.reduce_sum'
46:23: INFO: Renamed 'tf.squared_difference' to 'tf.math.squared_difference'
47:15: INFO: Changed tf.to_float call to tf.cast(..., dtype=tf.float32).
51:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
53:54: INFO: Renamed 'tf.layers.batch_normalization' to 'tf.compat.v1.layers.batch_normalization'
57:11: INFO: Changing keep_prob arg of tf.nn.dropout to rate, and recomputing value.

63:18: INFO: tf.random_normal_initializer requires manual check. Initializers no longer have the dtype argument in the constructor or partition_info argument in the __call__ method.
The calls have been converted to compat.v1 for safety (even though they may already have been correct).
63:18: INFO: Renamed 'tf.random_normal_initializer' to 'tf.compat.v1.random_normal_initializer'
64:20: INFO: Renamed 'tf.image.resize_images' to 'tf.image.resize'
65:14: INFO: Renamed 'tf.layers.separable_conv2d' to 'tf.compat.v1.layers.separable_conv2d'
72:11: INFO: Changing keep_prob arg of tf.nn.dropout to rate, and recomputing value.

77:14: ERROR: Using member tf.contrib.layers.conv2d_transpose in deprecated module tf.contrib. tf.contrib.layers.conv2d_transpose cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
80:62: INFO: Renamed 'tf.layers.batch_normalization' to 'tf.compat.v1.layers.batch_normalization'
84:11: INFO: Changing keep_prob arg of tf.nn.dropout to rate, and recomputing value.

87:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
89:54: INFO: Renamed 'tf.layers.batch_normalization' to 'tf.compat.v1.layers.batch_normalization'
95:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
102:14: ERROR: Using member tf.contrib.layers.conv2d in deprecated module tf.contrib. tf.contrib.layers.conv2d cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
109:11: INFO: Added keywords to args of function 'tf.nn.max_pool'
109:11: INFO: Renamed keyword argument for tf.nn.max_pool from value to input
109:11: INFO: Renamed 'tf.nn.max_pool' to 'tf.nn.max_pool2d'
127:20: WARNING: Using member tf.contrib.rnn.ConvLSTMCell in deprecated module tf.contrib.rnn. (Manual edit required) tf.contrib.rnn.* has been deprecated, and widely used cells/functions will be moved to tensorflow/addons repository. Please check it there and file Github issues if necessary.
127:20: ERROR: Using member tf.contrib.rnn.ConvLSTMCell in deprecated module tf.contrib. tf.contrib.rnn.ConvLSTMCell cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
138:13: INFO: Added keywords to args of function 'tf.shape'
141:14: INFO: Renamed 'tf.nn.dynamic_rnn' to 'tf.compat.v1.nn.dynamic_rnn'
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/sofilstm/nets_sofi.py'
 outputting to 'convLSTM_predicttimeseries_v2/sofilstm/nets_sofi.py'
--------------------------------------------------------------------------------

33:38: INFO: Added keywords to args of function 'tf.transpose'
35:24: INFO: Renamed 'tf.nn.dynamic_rnn' to 'tf.compat.v1.nn.dynamic_rnn'
38:19: INFO: Added keywords to args of function 'tf.transpose'
43:17: INFO: Renamed 'tf.variable_scope' to 'tf.compat.v1.variable_scope'
47:45: INFO: Added keywords to args of function 'tf.math.reduce_mean'
59:21: INFO: Added keywords to args of function 'tf.nn.conv2d'
59:21: INFO: Renamed keyword argument for tf.nn.conv2d from filter to filters
69:23: INFO: Added keywords to args of function 'tf.transpose'
73:24: INFO: Renamed 'tf.nn.dynamic_rnn' to 'tf.compat.v1.nn.dynamic_rnn'
76:19: INFO: Added keywords to args of function 'tf.transpose'
81:13: INFO: Renamed 'tf.variable_scope' to 'tf.compat.v1.variable_scope'
85:38: INFO: Added keywords to args of function 'tf.transpose'
92:37: INFO: Added keywords to args of function 'tf.math.reduce_mean'
99:25: INFO: Added keywords to args of function 'tf.nn.conv2d'
99:25: INFO: Renamed keyword argument for tf.nn.conv2d from filter to filters
106:4: INFO: tf.summary.image requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
106:4: INFO: Renamed 'tf.summary.image' to 'tf.compat.v1.summary.image'
107:4: INFO: tf.summary.image requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
107:4: INFO: Renamed 'tf.summary.image' to 'tf.compat.v1.summary.image'
108:4: INFO: tf.summary.image requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
108:4: INFO: Renamed 'tf.summary.image' to 'tf.compat.v1.summary.image'
109:4: INFO: tf.summary.image requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
109:4: INFO: Renamed 'tf.summary.image' to 'tf.compat.v1.summary.image'
110:4: INFO: tf.summary.image requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
110:4: INFO: Renamed 'tf.summary.image' to 'tf.compat.v1.summary.image'
110:89: INFO: Added keywords to args of function 'tf.math.reduce_mean'
111:4: INFO: tf.summary.image requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
111:4: INFO: Renamed 'tf.summary.image' to 'tf.compat.v1.summary.image'
121:9: INFO: Added keywords to args of function 'tf.reduce_min'
122:9: INFO: Added keywords to args of function 'tf.reduce_max'
125:12: INFO: Added keywords to args of function 'tf.shape'
126:12: INFO: Added keywords to args of function 'tf.shape'
128:8: INFO: Added keywords to args of function 'tf.transpose'
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/sofilstm/nets_sofi_old.py'
 outputting to 'convLSTM_predicttimeseries_v2/sofilstm/nets_sofi_old.py'
--------------------------------------------------------------------------------

44:9: INFO: Renamed 'tf.variable_scope' to 'tf.compat.v1.variable_scope'
48:23: INFO: Added keywords to args of function 'tf.transpose'
52:24: INFO: Renamed 'tf.nn.dynamic_rnn' to 'tf.compat.v1.nn.dynamic_rnn'
56:19: INFO: Added keywords to args of function 'tf.transpose'
61:13: INFO: Renamed 'tf.variable_scope' to 'tf.compat.v1.variable_scope'
71:4: INFO: tf.summary.image requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
71:4: INFO: Renamed 'tf.summary.image' to 'tf.compat.v1.summary.image'
72:4: INFO: tf.summary.image requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
72:4: INFO: Renamed 'tf.summary.image' to 'tf.compat.v1.summary.image'
73:4: INFO: tf.summary.image requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
73:4: INFO: Renamed 'tf.summary.image' to 'tf.compat.v1.summary.image'
74:4: INFO: tf.summary.image requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
74:4: INFO: Renamed 'tf.summary.image' to 'tf.compat.v1.summary.image'
74:74: INFO: Added keywords to args of function 'tf.math.reduce_mean'
75:4: INFO: tf.summary.image requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
75:4: INFO: Renamed 'tf.summary.image' to 'tf.compat.v1.summary.image'
75:73: INFO: Added keywords to args of function 'tf.math.reduce_mean'
85:9: INFO: Added keywords to args of function 'tf.reduce_min'
86:9: INFO: Added keywords to args of function 'tf.reduce_max'
89:12: INFO: Added keywords to args of function 'tf.shape'
90:12: INFO: Added keywords to args of function 'tf.shape'
92:8: INFO: Added keywords to args of function 'tf.transpose'
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/sofilstm/sofi.py'
 outputting to 'convLSTM_predicttimeseries_v2/sofilstm/sofi.py'
--------------------------------------------------------------------------------

44:8: INFO: Renamed 'tf.reset_default_graph' to 'tf.compat.v1.reset_default_graph'
53:17: INFO: Renamed 'tf.placeholder' to 'tf.compat.v1.placeholder'
54:17: INFO: Renamed 'tf.placeholder' to 'tf.compat.v1.placeholder'
55:21: INFO: Renamed 'tf.placeholder' to 'tf.compat.v1.placeholder'
56:25: INFO: Renamed 'tf.placeholder' to 'tf.compat.v1.placeholder'
59:18: INFO: Added keywords to args of function 'tf.shape'
60:18: INFO: Added keywords to args of function 'tf.shape'
61:28: INFO: Added keywords to args of function 'tf.shape'
90:23: INFO: Added keywords to args of function 'tf.reduce_mean'
104:19: INFO: tf.losses.mean_squared_error requires manual check. tf.losses have been replaced with object oriented versions in TF 2.0 and after. The loss function calls have been converted to compat.v1 for backward compatibility. Please update these calls to the TF 2.0 versions.
104:19: INFO: Renamed 'tf.losses.mean_squared_error' to 'tf.compat.v1.losses.mean_squared_error'
117:8: INFO: Renamed 'tf.train.write_graph' to 'tf.io.write_graph'
131:15: INFO: Renamed 'tf.global_variables_initializer' to 'tf.compat.v1.global_variables_initializer'
132:13: INFO: Renamed 'tf.Session' to 'tf.compat.v1.Session'
153:16: INFO: Renamed 'tf.train.Saver' to 'tf.compat.v1.train.Saver'
154:20: WARNING: *.save requires manual check. (This warning is only applicable if the code saves a tf.Keras model) Keras model.save now saves to the Tensorflow SavedModel format by default, instead of HDF5. To continue saving to HDF5, add the argument save_format='h5' to the save() function.
165:16: INFO: Renamed 'tf.train.Saver' to 'tf.compat.v1.train.Saver'
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/sofilstm/test_imageutil.py'
 outputting to 'convLSTM_predicttimeseries_v2/sofilstm/test_imageutil.py'
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/sofilstm/train.py'
 outputting to 'convLSTM_predicttimeseries_v2/sofilstm/train.py'
--------------------------------------------------------------------------------

60:38: INFO: tf.train.exponential_decay requires manual check. To use learning rate decay schedules with TensorFlow 2.0, switch to the schedules in `tf.keras.optimizers.schedules`.

60:38: INFO: Renamed 'tf.train.exponential_decay' to 'tf.compat.v1.train.exponential_decay'
66:25: INFO: Renamed 'tf.get_collection' to 'tf.compat.v1.get_collection'
66:43: INFO: Renamed 'tf.GraphKeys' to 'tf.compat.v1.GraphKeys'
68:28: INFO: Renamed 'tf.train.MomentumOptimizer' to 'tf.compat.v1.train.MomentumOptimizer'
75:25: INFO: Renamed 'tf.get_collection' to 'tf.compat.v1.get_collection'
75:43: INFO: Renamed 'tf.GraphKeys' to 'tf.compat.v1.GraphKeys'
77:28: INFO: Renamed 'tf.train.AdamOptimizer' to 'tf.compat.v1.train.AdamOptimizer'
88:26: INFO: tf.summary.merge_all requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
88:26: INFO: Renamed 'tf.summary.merge_all' to 'tf.compat.v1.summary.merge_all'
92:15: INFO: Renamed 'tf.global_variables_initializer' to 'tf.compat.v1.global_variables_initializer'
146:17: INFO: Renamed 'tf.ConfigProto' to 'tf.compat.v1.ConfigProto'
148:13: INFO: Renamed 'tf.Session' to 'tf.compat.v1.Session'
152:16: INFO: Renamed 'tf.train.write_graph' to 'tf.io.write_graph'
162:29: INFO: tf.summary.FileWriter requires manual check. The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.
162:29: INFO: Renamed 'tf.summary.FileWriter' to 'tf.compat.v1.summary.FileWriter'
207:20: WARNING: *.save requires manual check. (This warning is only applicable if the code saves a tf.Keras model) Keras model.save now saves to the Tensorflow SavedModel format by default, instead of HDF5. To continue saving to HDF5, add the argument save_format='h5' to the save() function.
209:28: WARNING: *.save requires manual check. (This warning is only applicable if the code saves a tf.Keras model) Keras model.save now saves to the Tensorflow SavedModel format by default, instead of HDF5. To continue saving to HDF5, add the argument save_format='h5' to the save() function.
256:16: INFO: Renamed 'tf.Summary' to 'tf.compat.v1.Summary'
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Processing file 'convLSTM_predicttimeseries/sofilstm/util.py'
 outputting to 'convLSTM_predicttimeseries_v2/sofilstm/util.py'
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------

