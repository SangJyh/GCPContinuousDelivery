# Dogcasso Generator
This application, hosted on Google Cloud, generates random images of dogs in the style of a Picasso painting. The transformation process begins with the training of a style transfer model. A painting, entitled [The Weeping Woman](https://en.wikipedia.org/wiki/The_Weeping_Woman]), is loaded from Wikipedia and fed into a [MobileNetV2](https://www.tensorflow.org/lite/models/style_transfer/overview) skeleton. This generates a 100x1x1 "style bottleneck" vector. Next, an image of a dog is retrieved using the [Dog Ceo random dog API](https://dog.ceo/dog-api/). The dog image and the style bottleneck are then input into a style transfer model that produces the final image. 

# Sample Output

![Sample Output](https://raw.githubusercontent.com/joekrinke15/GCPContinuousDelivery/main/dogpicasso.PNG)
