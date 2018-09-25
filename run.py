# -*- encoding: utf-8 -*-

from skinapp import app
#from skinapp.cocomodel import *

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=app.config['DEBUG'], port=app.config['PORT'])
