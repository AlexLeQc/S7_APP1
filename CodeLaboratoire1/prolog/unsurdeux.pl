un_sur_deux([]) :- true. 
un_sur_deux([_]) :- true.
un_sur_deux([_, X | Xs]) :- write(X), nl, un_sur_deux(Xs).

longueur([_|Qliste], NbItems) :- longueur(Qliste, NbItemsQueue), NbItems is NbItemsQueue + 1.