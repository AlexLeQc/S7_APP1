% Couleurs des serrures
serrure(bronze).
serrure(silver).
serrure(gold).

% Couleurs des cristaux
cristal(red).
cristal(blue).
cristal(yellow).
cristal(white).
cristal(black).

% Options de key
key(first).
key(second).
key(third).
key(fourth).
key(fifth).
key(sixth).

%Traduction nombre en text
value(first, 1).
value(second, 2).
value(third, 3).
value(fourth, 4).
value(fifth, 5).
value(sixth, 6).


% Trouver le nombre de truc dans une liste sans '' (Cette liste à un nombre de cristaux de X)
longueur([], 0).

longueur([_|Qliste], NombreItems):- 
	longueur(Qliste, NombreItemsQueue), NombreItems is NombreItemsQueue+1.

% Filtrer une liste en retirant les '' (Cette liste filtrer donne cette liste)
filtrer([], []).

filtrer([''|Qliste], ListeFiltree):-
    filtrer(Qliste, ListeFiltree), !.

filtrer([X|Qliste], [X|ListeFiltree]):-
    filtrer(Qliste, ListeFiltree).

% Compte le nombre de fois qu'un cristal est dans une liste
compter_occurences(Cristaux, Cristal, NbCristaux) :-
    findall(Cristal, (member(Cristal, Cristaux)), Liste),
    longueur(Liste, NbCristaux).

% Trouver a qu'elle position est le dernier cristal d'une certaine couleur (Aide de AI)
dernier_cristal_couleur(Cristaux, Cristal, Position) :-
    findall(P, nth1(P, Cristaux, Cristal), Positions),
    last(Positions, Position).

%Trouver la cle de la porte (Avec cette liste, on resoue la porte avec cette cle)
resoudre_porte([Serrure|CristauxNonFiltree], Cle) :-
    filtrer(CristauxNonFiltree, Cristaux),
    longueur(Cristaux, NbCristaux),
    resoudre_selon_nombre(Serrure, Cristaux, NbCristaux, Cle).

%Règles pour 3 cristaux
resoudre_selon_nombre(_, Cristaux, 3, second) :-
    \+ member(red, Cristaux), !.

resoudre_selon_nombre(_, Cristaux, 3, third) :-
    last(Cristaux, white), !.

resoudre_selon_nombre(_, Cristaux, 3, Cle) :-
    compter_occurences(Cristaux, blue, NbBlue), NbBlue >= 2,
    dernier_cristal_couleur(Cristaux, blue, Position),
    value(Cle, Position), !.

resoudre_selon_nombre(_, _, 3, first).

%Règles pour 4 cristaux
resoudre_selon_nombre(silver, Cristaux, 4, Cle) :-
    compter_occurences(Cristaux, red, NbRed), NbRed > 1,
    dernier_cristal_couleur(Cristaux, red, Position),
    value(Cle, Position), !.

resoudre_selon_nombre(_, Cristaux, 4, first) :-
    last(Cristaux, yellow),
    \+ member(red, Cristaux), !.

resoudre_selon_nombre(_, Cristaux, 4, first) :-
    compter_occurences(Cristaux, blue, NbBlue), NbBlue = 1, !.

resoudre_selon_nombre(_, Cristaux, 4, fourth) :-
    compter_occurences(Cristaux, yellow, NbYellow), NbYellow > 1, !.

resoudre_selon_nombre(_, _, 4, second).

%Règles pour 5 cristaux
resoudre_selon_nombre(gold, Cristaux, 5, fourth) :-
    last(Cristaux, black), !.

resoudre_selon_nombre(_, Cristaux, 5, first) :-
    compter_occurences(Cristaux, red, NbRed), NbRed = 1,
    compter_occurences(Cristaux, yellow, NbYellow), NbYellow > 1, !.

resoudre_selon_nombre(_, Cristaux, 5, second) :-
    \+ member(black, Cristaux), !.

resoudre_selon_nombre(_, _, 5, first).

%Règles pour 6 cristaux
resoudre_selon_nombre(bronze, Cristaux, 6, third) :-
    \+ member(yellow, Cristaux), !.

resoudre_selon_nombre(_, Cristaux, 6, fourth) :-
    compter_occurences(Cristaux, yellow, NbYellow), NbYellow = 1,
    compter_occurences(Cristaux, white, NbWhite), NbWhite > 1, !.

resoudre_selon_nombre(_, Cristaux, 6, sixth) :-
    \+ member(red, Cristaux), !.

resoudre_selon_nombre(_, _, 6, fourth).


    
    




    






