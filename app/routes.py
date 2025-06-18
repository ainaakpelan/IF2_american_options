from flask import Blueprint, render_template

routes = Blueprint("routes", __name__)

@routes.route("/")
def index():
    return render_template("index.html")

@routes.route("/bjerksund_stensland")
@routes.route("/bjerksund_stensland/<section>")
def bjerksund_stensland(section="theory"):
    return render_template("bjerksund_stensland.html", section=section)

@routes.route("/longstaff_schwartz")
@routes.route("/longstaff_schwartz/<section>")
def longstaff_schwartz(section="theory"):
    return render_template("longstaff_schwartz.html", section=section)

@routes.route("/stochastic_mesh")
@routes.route("/stochastic_mesh/<section>")
def stochastic_mesh(section="theory"):
    return render_template("stochastic_mesh.html", section=section)

@routes.route("/comparison")
def comparison():
    return render_template("comparison.html")
