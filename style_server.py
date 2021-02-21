'''
Title: style_server.py 
Author: Nick Johnson
Date created: 2/9/2021
Last modified: 2/21/2021 
Description: redis and keras server processes for artistic style transfer implementation
(a work in progress, exercise in understanding and explicating ai web services)

**References**
[Server implementation adapted from Adrian Rosebrock](
	https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/)

[Style transfer algorithm with code from a fchollet example](
	https://twitter.com/fchollet)

[Algorithm implemented w/ keras: A Neural Algorithm of Artistic Style](
	http://arxiv.org/abs/1508.06576)
'''


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from threading import Thread
from PIL import Image
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import sys
import io


"""
## Global constants
"""

# server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# Weights of the different loss components
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

# Dimensions of the generated picture.
width, height = keras.preprocessing.image.load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"


'''
## initializing model & redis / flask servers
'''


app = flask.Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None


"""
## Image preprocessing / deprocessing utilities
"""


def base64_encode_image(a):
	# base64 encode the input NumPy array
	return base64.b64encode(a).decode("utf-8")

def base64_decode_image(a, dtype, shape):
	# if this is Python 3, we need the extra step of encoding the
	# serialized NumPy string as a byte object
	if sys.version_info.major == 3:
		a = bytes(a, encoding="utf-8")
	# convert the string to a NumPy array using the supplied data
	# type and target shape
	a = np.frombuffer(base64.decodestring(a), dtype=dtype)
	a = a.reshape(shape)
	# return the decoded image
	return a

def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # how to handle/change image loading differences..
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    # if the image mode is not RGB, convert it
    #       if image.mode != "RGB":
    #               image = image.convert("RGB")
    return tf.convert_to_tensor(img)

def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


'''
## image network utilities
'''


# The gram matrix of an image tensor (feature-wise outer product)

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


# The "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


# An auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


# The 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent

def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


# List of layers to use for the style loss.
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]


# The layer to use for the content loss.
content_layer_name = "block5_conv2"


def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )

    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss



@tf.function # compiling loss and gradiat function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads



'''
## start filling flask from redis store for processing
'''

 
def generator():
                
        # Build a VGG19 model loaded with pre-trained ImageNet weights
        model = vgg19.VGG19(weights="imagenet", include_top=False)

        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        # Set up a model that returns the activation values for every layer in
        # VGG19 (as a dict).
        feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

        # can likely be moved outside of this function but should 
        optimizer = keras.optimizers.SGD(
                keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
                )
        )
        
	# listen for images to process
	while True:
		# attempt to grab a batch of images from the database, then
		# initialize the image IDs and batch of images themselves
		queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
		imageIDs = []
		batch = None
		# loop over the queue
		for q, r, s in queue:
			# deserialize the object and obtain the input image
			q = json.loads(q.decode("utf-8"))
                        r = json.loads(r.decode("utf-8"))
                        s = json.loads(s.decode("utf-8"))
                        
                        ##-->> change image contasts to style server congruent
			base_image = base64_decode_image(q["image"], IMAGE_DTYPE,
				(1, height, width, IMAGE_CHANS))
                        style_reference_image = base64_decode_image(r["image"], IMAGE_DTYPE,
				(1, height, width, IMAGE_CHANS))
                        combination_image = base64_decode_image(s["image"], IMAGE_DTYPE,
				(1, height, width, IMAGE_CHANS))
                                                
			# check to see if the batch list is None
			if batch is None:
				batch = image
			# otherwise, stack the data
			else:
				batch = np.vstack([batch, image])  # REVIEW NP STACK
			# update the list of image IDs
			imageIDs.append(q["id"])

		# check to see if we need to generate the batch
		if len(imageIDs) > 0:
			# classify the batch
			print("* Batch size: {}".format(batch.shape))

                        ### return a picture from generate() funcion
                        ###  we are returning pictures, not predictions..
                        ## results = model.generate(batch)
                        iterations = 4000
                        for i in range(1, iterations + 1):
                                loss, grads = compute_loss_and_grads(
                                        combination_image, base_image, style_reference_image
                                )
                                optimizer.apply_gradients([(grads, combination_image)])
                                if i % 100 == 0:
                                        print("Iteration %d: loss=%.2f" % (i, loss))
                                        img = deprocess_image(combination_image.numpy())
                                        fname = result_prefix + "_at_iteration_%d.png" % i
                                        # keras.preprocessing.image.save_img(fname, img)
                        

			# loop over the image IDs and their corresponding set of
			# results from our model
			for (imageID, resultSet) in zip(imageIDs, results):
				# initialize the list of output predictions
				output = []
				# loop over the results and add them to the list of
				# output predictions
				for (imagenetID, label, prob) in resultSet:
					r = {"label": label, "probability": float(prob)}
					output.append(r)
				# store the output predictions in the database, using
				# the image ID as the key so we can fetch the results
				db.set(imageID, json.dumps(output))
			# remove the set of images from our queue
			db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)
		# sleep for a small amount
		time.sleep(SERVER_SLEEP)

        

@app.route("/generate", methods=["POST"])
def generate():
        # generte is used inside of read_image() to gerate the style transfered output
        # initialize the data dictionary that will be returned from the view
	data = {"success": False}

        # if image uploaded to endpoint
        # need to edit for two-image-input?
	if flask.request.method == "POST":  # handling both base and style ref imgs?
		if flask.request.files.get("image"): 
			# read the images in PIL format and preprocess
                        base_image = preprocess_image(base_image_path)
                        style_reference_image = preprocess_image(style_reference_image_path)
                        combination_image = tf.Variable(preprocess_image(base_image_path))
                        
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = image.copy(order="C")
			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			d = {"id": k, "image": base64_encode_image(image)}
			db.rpush(IMAGE_QUEUE, json.dumps(d))

			# keep looping until our model server returns the output
			# predictions
			while True:
				# attempt to grab the output predictions
				output = db.get(k)
				# check to see if our model has classified the input
				# image
				if output is not None:
 					# add the output predictions to our data
 					# dictionary so we can return it to the client
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)
					# delete the result from the database and break
					# from the polling loop
					db.delete(k)
					break
				# sleep for a small amount to give the model a chance
				# to classify the input image
				time.sleep(CLIENT_SLEEP)
			# indicate that the request was a success
			data["success"] = True
	# return the data dictionary as a JSON response
	return flask.jsonify(data)
 

'''
## load the model, start the server
'''

if __name__ == "__main__":
	# load the function used to classify input images in a *separate*
	# thread than the one used for main classification
	print("* Starting model service...")
	t = Thread(target=generator, args=())
	t.daemon = True
	t.start()
	# start the web server
	print("* Starting web service...")
	app.run()
 

        
                        

                        
