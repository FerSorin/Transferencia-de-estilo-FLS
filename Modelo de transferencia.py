import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from PIL import Image
import time
import functools
import tensorflow as tf

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

print(tf.__version__)

# Estuve leyendo y aprendiendo sobre "eager execution", que es un modo de operación que 
# permite la evaluación inmediata de las operaciones de TensorFlow en lugar de construir primero grafos computacionales y luego ejecutarlos en una sesión
try:
    tf.enable_eager_execution()
except Exception:
  pass

print("Eager execution: {}".format(tf.executing_eagerly()))

content_path = '../figs/lion.jpg'      # Aquí se coloca la ruta a la imagen de contenido
style_path = '../figs/violin.jpg'      # Aquí ponemos la ruta a la imagen de estilo

#	Creamos dos secciones, unaa que cargará las imágenes proporcionadas
#	y la otra que nos las mostrará
In [4]:
def load_img(path_to_img):
  max_dim = 512
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
   
  img = kp_image.img_to_array(img)

  # Establecemos una matriz de imágenes que tenga excatamente las mismas dimensiones que el batch
  img = np.expand_dims(img, axis=0)
  return img
In [5]:
def imshow(img, title=None):
    
  # Eliminamos la dimension del batch
  out = np.squeeze(img, axis=0)
    
  # Normalizamos para poder visualizar 
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)

#Visualizamos la imagenes
plt.figure(figsize=(10,10))

content = load_img(content_path).astype('uint8')
style = load_img(style_path).astype('uint8')

plt.subplot(1, 2, 1)
imshow(content, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style, 'Style Image')
plt.show()

def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("La imagen debe de tener dimensiones de "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Input no valido")
  
  # Realizamos el inverso del paso de preprocesamiento
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x
# Capa de contenido donde extraeremos nuestros mapas de características
content_layers = ['block5_conv2'] 

# Capas de estilo que nos interesan
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def get_model():
#  """Creamos nuestro modelo con acceso a capas intermedias.
#  
#   Esta función cargará el modelo y accederá a las capas intermedias.
#   Estas capas se usan para crear un nuevo modelo que tomará la imagen de entrada
#   y devolver los resultados de estas capas intermedias del modelo.
#     Devuelve un modelo de Keras que toma entradas de imagen y emite el estilo y
#       contenido de capas intermedias.
#  """
  # Carguamos nuestro modelo. Cargamos el modelo preentrenado con los datos de ImageNet
  
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
    
  # Obtenemos las capas de salida correspondientes a las capas de estilo y contenido
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
    
  # Construimos el modelo 
  return models.Model(vgg.input, model_outputs)
def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))
def gram_matrix(input_tensor):
  
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):

  
  # Altura, anchura y número de filtros de cada capa
  # Escalamos la pérdida en una capa dada por el tamaño del mapa de características y la cantidad de filtros

  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  
  return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (canales ** 2) * (ancho * alto) ** 2)

def get_feature_representations(model, content_path, style_path):

 
  # Cargamos nuestras imágenes

  content_image = load_and_process_img(content_path)
  style_image = load_and_process_img(style_path)
  
  # Computa outputs

  style_outputs = model(style_image)
  content_outputs = model(content_image)
  
  
  # Obtiene las representaciones de estilo y contenido de nuestro modelo 

  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
  return style_features, content_features

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):

  style_weight, content_weight = loss_weights
  
  #Ingresamos nuestra imagen a nuestro modelo. Esto nos dará el contenido y
  # las representaciones de estilo en nuestras capas deseadas. 

  model_outputs = model(init_image)
  
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]
  
  style_score = 0
  content_score = 0

  # Acumulación de pérdidas de estilo de todas las capas
  # Aquí ponderamos igual cada contribución de cada capa de pérdida

  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
    
  # Calculamos los gradientes de las imágenes de input

  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss
In [17]:
import IPython.display

def run_style_transfer(content_path, 
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3, 
                       style_weight=1e-2): 
    
  # No necesitamos,ni queremos entrenar ninguna capa del modelo, por lo que establecemos 
  #su entrenamiento en falso
  model = get_model() 
  for layer in model.layers:
    layer.trainable = False
  
  # Obtenemmos las representaciones de características de estilo y contenido 
  # (de nuestras capas intermedias especificadas) 
  style_features, content_features = get_feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
  # Establecemos imagen inicial
  init_image = load_and_process_img(content_path)
  init_image = tf.Variable(init_image, dtype=tf.float32)
  # Create our optimizer
  opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

  # Para las visualización de las imágenes intermedias. Durante el proceso veremos como va cambiando la imagen
  iter_count = 1
  
  # Guardamos el mejor resultado
  best_loss, best_img = float('inf'), None
  
  # Creamos una configuración para que sea "agradable" verlo
  loss_weights = (style_weight, content_weight)
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
  }
    
  # Para la visualización
  num_rows = 2
  num_cols = 5
  display_interval = num_iterations/(num_rows*num_cols)
  start_time = time.time()
  global_start = time.time()
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  
  imgs = []
  for i in range(num_iterations):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time() 
    
    if loss < best_loss:
      # Actualizamos la mejor pérdida y la mejor imagen de la pérdida total 
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

    if i % display_interval== 0:
      start_time = time.time()
      
      # Usamos el método .numpy () para obtener la matriz numpy concreta
      plot_img = init_image.numpy()
      plot_img = deprocess_img(plot_img)
      imgs.append(plot_img)
      IPython.display.clear_output(wait=True)
      IPython.display.display_png(Image.fromarray(plot_img))
      print('Iteration: {}'.format(i))        
      print('Total loss: {:.4e}, ' 
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '
            'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
  print('Total time: {:.4f}s'.format(time.time() - global_start))
  IPython.display.clear_output(wait=True)
  plt.figure(figsize=(14,4))
  for i,img in enumerate(imgs):
      plt.subplot(num_rows,num_cols,i+1)
      plt.imshow(img)
      plt.xticks([])
      plt.yticks([])
      
  return best_img, best_loss
Visualizamos la imagen de contenido, la imagen de estilo y la imagen final.
In [19]:
def show_results(best_img, content_path, style_path, show_large_final=True):
  plt.figure(figsize=(10, 5))
  content = load_img(content_path) 
  style = load_img(style_path)

  plt.subplot(1, 2, 1)
  imshow(content, 'Content Image')

  plt.subplot(1, 2, 2)
  imshow(style, 'Style Image')

  if show_large_final: 
    plt.figure(figsize=(10, 10))

    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()
In [20]:
show_results(best, content_path, style_path)

