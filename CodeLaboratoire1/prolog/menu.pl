horsdoeuvre(salade, 1).
horsdoeuvre(pate, 6).

poisson(sole, 2).
poisson(thon, 4).

viande(porc, 7).
viande(boeuf, 3).

dessert(glace, 5).
dessert(fruit, 1).

repas(H, P, D) :- horsdoeuvre(H, _), (poisson(P, _) ; viande(P, _)), dessert(D, _).

repasLeger(H, P, D) :- horsdoeuvre(H, PointH), (poisson(P, PointP) ; viande(P, PointP)), dessert(D, PointD), PointTotal is PointH + PointP + PointD, PointTotal < 10.


