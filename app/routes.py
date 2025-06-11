from flask import Blueprint, render_template
from library.models.bjerksund_stensland import BjerksundStensland
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

routes = Blueprint("routes", __name__)

@routes.route("/")
def index():
    return render_template("base.html")

@routes.route("/bs_model")
def bs_model():
    S = 100
    r = 0.05
    q = 0.02
    b = r - q  # cost of carry
    T = 1
    sigma = 0.2
    option_type = "call"

    strikes = np.linspace(50, 150, 100)
    prices = [
        BjerksundStensland(S=S, K=K, T=T, r=r, b=b, sigma=sigma, option_type=option_type).calculate_price()
        for K in strikes
    ]

    fig, ax = plt.subplots()
    ax.plot(strikes, prices)
    ax.set_title("Cena opcji wg metody Bjerksund-Stensland")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Cena opcji")
    ax.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")

    return render_template("bs_model.html", plot_url=plot_url)

@routes.route("/method2")
def method2():
    message = "To jest metoda 2 – tutaj pojawi się inna metoda wyceny."
    return render_template("method2.html", message=message)

@routes.route("/method3")
def method3():
    message = "To jest metoda 3 – tutaj pojawi się jeszcze inna metoda wyceny."
    return render_template("method3.html", message=message)
