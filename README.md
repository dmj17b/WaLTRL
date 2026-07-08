# WaLTRL
Deep Reinforcement Learning Repo for WaLTER Robot



# Brax training patch:
Latest JAX updates seem to have broken some Brax code. As of now, my fix is to manually edit the file: 
```/site-packages/brax/training/agents/ppo/train.py ```

and change line 756 from 

```   
training_state = jax.device_put_replicated(
       training_state, jax.local_devices()[:local_devices_to_use]
   )
```

to: 
```
  devices = jax.local_devices()[:local_devices_to_use]
  # 1. Define a 1D logical mesh of your devices
  mesh = jax.sharding.Mesh(devices, ('batch',))
  # 2. Define a NamedSharding that splits data across the 'batch' axis
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))

  # 3. Broadcast the training state to all devices and apply the sharding
  training_state = jax.tree_util.tree_map(
      lambda x: jax.device_put(jax.numpy.broadcast_to(x, (len(devices),) + jax.numpy.shape(x)), sharding),
      training_state
  )
```