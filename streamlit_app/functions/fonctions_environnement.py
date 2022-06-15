from psutil import virtual_memory, cpu_percent
import tensorflow as tf

def visualiation_RAM():
  ram_gb = virtual_memory().total / 1e9
  ram_prc = virtual_memory().percent
  ram_used = virtual_memory().used / 1e9
  ram_free = virtual_memory().free / 1e9
  print('Utilisation CPU : ', cpu_percent(4), '%') 
  print('RAM disponnible {:.1f} gigabytes'.format(ram_gb))
  print('utilisation de la RAM disponnible : {} % '.format(ram_prc))
  print('RAM utilisée {:.1f} gigabytes'.format(ram_used))
  print('RAM libre {:.1f} gigabytes of free RAM'.format(ram_free))
  
  
# def info_pgu_nvidia():
#   gpu_info = !nvidia-smi
#   gpu_info = '\n'.join(gpu_info)
#   if gpu_info.find('failed') >= 0:
#       print('Non-connecté à un GPU')
#   else:
#       print(gpu_info)

def info_cpu_gpu():
  print("liste des cpu et gpu détectés :")
  tf.config.list_physical_devices()
  



visualiation_RAM()
info_cpu_gpu()