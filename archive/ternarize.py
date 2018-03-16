import tensorflow as tf
import os, glob
import numpy as np

# changing the trained weights from floats into ternary
src_weights_dir = "experiments/base_model/best_weights"
dest_weights_dir = "experiments/ternary/best_weights"
threshold = 0.7

meta_src = os.path.join(src_weights_dir, "after-epoch-10.meta")
meta_dest = os.path.join(dest_weights_dir, "ternary")

# load/restore the weights
sess = tf.Session()
saver = tf.train.import_meta_graph(meta_src)
saver.restore(sess, tf.train.latest_checkpoint(src_weights_dir))

# ternarize the weights
print("Ternarizing the weights")

graph = tf.get_default_graph()

#for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
	print("Altering weights for %s" %variable.name)
	w_tensor = sess.run(variable.name)

	# finding the mean of the average
	mean = np.mean(np.abs(w_tensor))
	print(threshold * mean)

	pos_tensor = np.copy(w_tensor)
	neg_tensor = np.copy(w_tensor)

	pos_tensor[pos_tensor > threshold * mean] = 1.0
	pos_tensor[pos_tensor != 1.0] = 0.0
	neg_tensor[neg_tensor < (-1.0) * threshold * mean] = -1.0
	neg_tensor[neg_tensor != 1.0] = 0.0

	final_tensor = np.add(pos_tensor, neg_tensor)
	print(w_tensor)
	print(final_tensor)

	assign_op = variable.assign(final_tensor)
	sess.run(assign_op)


# save the weights
save_path = saver.save(sess, meta_dest)
print("Saving to %s" %save_path)

sess.close()