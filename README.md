Homomorphic Encryption Testing
==============================


![A black and white image of an astronaut](/astro_256.png)
![Edges highlighted from the previous image](/astro_256_edges.png)

This repo uses the [Pyfhel](https://pyfhel.readthedocs.io/en/latest/) module 
(which uses [Microsoft SEAL](https://github.com/microsoft/SEAL))
to implement an edge detection algorithm that runs on an 
**encrypted** image.

Microsoft SEAL has a similar [demo](https://github.com/microsoft/EVA/blob/main/examples/image_processing.py) using their python-based compiler named EVA.

Unlike that example though here we implement full separation between the trusted client and the untrusted server. We also parallelize the encryption algorithm which is the most expensive part of the algorithm.

## Performance

As a baseline, the edge detection algorithm ran in in 200ms for a 512x512 unencrypted image. The encryption algorithm took 15 minutes (!) for the 512x512 and failed to complete because it occupied more than 28GB of RAM. Part of the problem is that since we have a client and server speaking JSON, the large encrypted data set is duplicated multiple times.

After implementing streaming of responses, I was able to complete a full process for 256x256 image. The encryption process took 200 seconds (using 12 cores) and the rest of the algorithm and decryption took 60 seconds. The RAM usage was around 12GB.