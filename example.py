from model.gardener_model import VirtualGardener

gardener = VirtualGardener('./predicts/')
res1 = gardener.predict('./plants/PotatoHealthy2.JPG')