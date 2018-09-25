from flask import Flask

app = Flask(__name__)
app.config.from_object('skinapp.settings')

import skinapp.views
