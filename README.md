# AI_API
Restful API using redis data store to broker requests for keras  

Currently working with Python 3.6.9
I have included a dependency file to source with pip

    pip install -r requirements.txt
    
Most of the image style transfer algorithm is put together.
Still flushing out redis and flask processes.
Will need further stitching of constants and datatypes between.

Using redis to handle a large queue of images to render in the style of another image (or styles of a queue of images),
this proto web service might be used to retexture videogames in the style of another game/artist/etc.
Or provide the backend for scaling other applications of artistic style transfer.

[A Neural Algorithm of Artistic Style] - http://arxiv.org/abs/1508.06576
