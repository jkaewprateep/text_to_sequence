# text_to_sequence
Data input from text and public sources for AI and sequence working process.

```
input_word = tf.constant(' \'Cause it\'s easy as an ice cream sundae Slipping outta your hand into the dirt Easy as an ice cream sundae Every dancer gets a little hurt Easy as an ice cream sundae Slipping outta your hand into the dirt Easy as an ice cream sundae Every dancer gets a little hurt Easy as an ice cream sundae Oh, easy as an ice cream sundae ')
vocab = [ "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_", 
"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
layer = tf.keras.layers.StringLookup(vocabulary=vocab)
sequences_mapping_string = layer(tf.strings.bytes_split(input_word))
```

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Method 1 create label from map it with vocaburary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print( 'input_word: ' + str(input_word) )
print( " " )
print( tf.strings.bytes_split(input_word) )
print( sequences_mapping_string )
```

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Method 2 create label from it tokenizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
text = "Cause its easy as an ice cream sundae Slipping outta your hand"
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='oov', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,)
tokenizer.fit_on_texts([text])
```

```
i_count = tf.strings.split([text])[0].shape[0] + 1
aDict = json.loads(tokenizer.to_json())
text_input = tf.constant([''], shape=())

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def auto_paddings( data, max_sequences=15 ):
	data = tf.constant( data, shape=(data.shape[0], 1) )
	paddings = tf.constant([[1, 15 - data.shape[0] - 1], [0, 0]])
	padd_data = tf.pad( data, paddings, "CONSTANT" )
	padd_data = tf.constant( padd_data, shape=(15, 1) ).numpy()
	return padd_data


input_word = tf.zeros([1, 15, 1], dtype=tf.int64)
input_label = tf.ones([1, 1, 1], dtype=tf.int64)

for i in range(i_count):
	word = json.loads(aDict['config']['index_word'])[str(i + 1)]
	i_word = layer(tf.strings.bytes_split(word))
	padd_data = tf.constant(auto_paddings( i_word, 15 ), shape=(1, 15, 1))
	
	index = json.loads(aDict['config']['word_index'])[word]

	if i > 0:
		input_word = tf.experimental.numpy.vstack([input_word, padd_data])
		input_label = tf.experimental.numpy.vstack([input_label, tf.constant(index, shape=(1, 1, 1))])


dataset = tf.data.Dataset.from_tensors(( input_word, input_label ))
```

![ice-cream](https://github.com/jkaewprateep/text_to_sequence/blob/main/images.jpg "ice-cream") 
![ice-cream](https://github.com/jkaewprateep/text_to_sequence/blob/main/image3.jpg "ice-cream")
![ice-cream](https://github.com/jkaewprateep/text_to_sequence/blob/main/image4.jpg "ice-cream")
