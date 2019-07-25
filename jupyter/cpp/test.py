import location_nn

tf = location_nn.TemplateFilterer()
tf.add('a',['b','c'])
print(tf.search(['a', 'b']))
