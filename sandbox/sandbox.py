import tensorflow as tf

v = tf.Variable([1, 2])
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    # Usage passing the session explicitly.
    print v.eval(sess)
    # Usage with the default session.  The 'with' block
    # above makes 'sess' the default session.
    print v.eval()
