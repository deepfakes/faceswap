# deepfakes_faceswap
This is the code from [deepfakes' faceswap project](https://www.reddit.com/user/deepfakes/).
Hope we can improve it together, HAVE FUN!

Message from deepfakes:

**Whole project with training images and trained model (~300MB):**  
anonfile.com/p7w3m0d5be/face-swap.zip or [click here to download](anonfile.com/p7w3m0d5be/face-swap.zip)

**Source code only:**  
anonfile.com/f6wbmfd2b2/face-swap-code.zip or [click here to download](anonfile.com/f6wbmfd2b2/face-swap-code.zip)

**Requirements:**

    Python 3
    Opencv 3
    Tensorflow 1.3+(?)
    Keras 2

you also need a modern GPU with CUDA support for best performance

**How to run:**

    python train.py

As you can see, the code is embarrassingly simple. I don't think it's worth the trouble to keep it secret from everyone.
I believe the community are smart enough to finish the rest of the owl.

If there is any question, welcome to discuss here.

**Some tips:**

Reuse existing models will train much faster than start from nothing.  
If there are not enough training data, start with someone looks similar, then switch the data.
