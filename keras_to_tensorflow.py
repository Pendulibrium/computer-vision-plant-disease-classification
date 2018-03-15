from keras.models import load_model
from keras import backend as K
import tensorflow as tf


# Loading model in keras
model_name = "2018-03-07_20:51:35-2018-03-07_23:02:18"
model_path = "trained_models/"+model_name+".h5"
model = load_model(model_path)

# Get current session using the keras backend (Tensorflow)
sess = K.get_session()

checkpoint_folder = "checkpoints/"
saver =  tf.train.Saver()
# Saving checkpoing a.k.a saving the weights of the model
saver.save(sess, checkpoint_folder + "checkpoint-1.ckpt")
# Saving the graph of the model
tf.train.write_graph(sess.graph, logdir=checkpoint_folder, name="graph_1.pb", as_text=False)


