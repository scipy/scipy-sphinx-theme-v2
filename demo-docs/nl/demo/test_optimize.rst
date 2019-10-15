Optimalisatie (:mod:`scipy.optimize`)
============================================

.. sectionauthor:: Travis E. Oliphant

.. sectionauthor:: Pauli Virtanen

.. sectionauthor:: Denis Laxalde

.. currentmodule:: scipy.optimaliseren

Het :mod:`scipy.optimize` pakket biedt meerdere algemeen gebruikt
optimalisatiealgoritmen. Een gedetailleerde lijst is beschikbaar:
:mod:`scipy.optimize` (kan ook worden gevonden door ``help(scipy.optimize)``).

De module bevat:

1. Onbeperkt en beperkt minimaliseren van multivariate schaalr
   functies (:func:`minimize`) met behulp van een verscheidenheid aan algoritmen (bijv. BFGS,
   Nelder-Mead simplex, Newton Conjugate Gradient, COBYLA of SLSQP)

2. Globale (brute-force) optimalisatieroutes (bijv., :func:`anneal`, :func:`basinhopping`)

3. Minst-squares minimalisering (:func:`leastsq`) en curve fit
   (:func:`curve_fit`) algoritmen

4. Scalar universiteitsfuncties minimaliseren (:func:`minimalis_scalar`) en
   root finderen (:func:`newton`)

5. Multivariate systeem sollicvers (:func:`root`) met behulp van een verscheidenheid van
   algoritmen (bijv. hybride Powell, Levenberg-Marquardt of grootschalige
   methoden zoals Newton-Krylov [KK]_).

Onder de andere tonen verschillende voorbeelden aan dat ze een fundamenteel gebruik hebben.


Onvoorwaardelijke minimalisering van multivariate schaalfuncties (:func:`minimaliseren`)
------------------------------------------------------------------------------

De :func:`minimalis` functie biedt een gemeenschappelijke interface naar niet-gebonden
en beperkt geminimaliseerde algoritmen voor multivariate schaalyses
in `scipy.optimize`. Om de minimization functie te tonen, kijk naar de
probleem van het minimaliseren van de Rosenbrock functie van :math:`N` variabelen:

.. wiskunde::
   :nowrap:

    \[ f\left(\mathbf{x}\right)=\sum_{i=1} ^{N-1}100\left(x_{i}-x_{i-1}^{2}\right) ^{2}+\left(1-x_{i-1}\right) ^{2}\]

De minimale waarde van deze functie is 0 die bereikt wordt wanneer
:math:`x_{i}=1.`

Let op dat de Rosenbrock functie en de derivaten erin zijn opgenomen
`scipy.optimize`. De implementaties die in de volgende secties worden getoond
geef voorbeelden van hoe een objectieve functie en de functie ervan gedefinieerd moeten worden
jacobiaanse en hessiaanse functies.

Nelder-Mead Simplex algoritme (``method='Nelder-Mead'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^ ^ ^

In het onderstaande voorbeeld wordt de :func:`minimalis` routine gebruikt
met het *Nelder-Mead* simplex algoritme (geselecteerd via de ``method``
parameter):

    >>> Importeer numpy als np
    >>> van scipy.geoptimaliseerd import minimaliseren

    >>> rozen (x) ontdooien:
    ...     """De Rosenbrock functie""""
    ...     retoursom(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    >>> x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    >>> res = minimaliseren(rosen, x0, methode='nelder-mead',
    ...                opties={'xtol': 1e-8, 'disp': True})
    Optimalisatie succesvol beëindigd.
             Huidige functie waarde: 0.000000
             Iteraties: 339
             Functiebeoordelingen: 571

    >>> print(res.x)
    [ 1. 1. 1. 1. 1. 1.]

Het simplex algoritme is waarschijnlijk de eenvoudigste manier om een eerlijk te minimaliseren
goed gedragen functie. Het vereist alleen beoordelingen van functies en is een goed
keuze voor eenvoudige minimaliseringsproblemen. Maar omdat het niet gebruikt wordt
eventuele graduatie, het kan langer duren om het minimum te vinden.

Een ander optimalisatiealgoritme dat alleen functies nodig heeft om te vinden
het minimum is *Powell*'s methode beschikbaar door ``method='powell'`` in te stellen
:func:`minimaliseren`.


Broyden-Fletcher-Goldfarb-Shanno algoritme (``method='BFGS'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^ ^ ^

Om sneller te kunnen convergeren naar de oplossing, gebruikt deze routine
de graduatie van de objectieve functie. Als de gradient niet is gegeven
door de gebruiker, dan wordt het geschat met eerste verschillen. De
Broyden-Fletcher-Goldfarb-Shanno (BFGS) methode vereist meestal
minder functies oproepen dan het simplex algoritme zelfs wanneer de gradient
moet worden ingeschat.

Om dit algoritme te laten zien wordt de Rosenbrock functie opnieuw gebruikt.
De gradient van de Rosenbrock functie is de vector:

.. wiskunde::
   :nowrap:

    \begin{eqnarray*} \frac{\partial f}{\partial x_{j}} & = & \sum_{i=1}^{N}200\left(x_{i}-x_{i-1}^{2}\right)\left(\delta_{i,j}-2x_{i-1}\delta_{i-1,j}\right)-2\left(1-x_{i-1}\right)\delta_{i-1,j}.\\  & = & 200\left(x_{j}-x_{j-1}^{2}\right)-400x_{j}\left(x_{j+1}-x_{j}^{2}\right)-2\left(1-x_{j}\right).\end{eqnarray*}

Deze uitdrukking is geldig voor de interieur-derivaten. Speciale gevallen
zijn

.. wiskunde::
   :nowrap:

    \begin{eqnarray*} \frac{\partial f}{\partial x_{0}} & = & -400x_{0}\left(x_{1}-x_{0}^{2}\right)-2\left(1-x_{0}\right),\\ \frac{\partial f}{\partial x_{N-1}} & = & 200\left(x_{N-1}-x_{N-2}^{2}\right).\end{eqnarray*}

Een Python functie die deze gradient berekent wordt gemaakt door de
code-segment:

    >>> rosen_der(x) ontdooien:
    ...     xm = x[1:-1]
    ...     xm_m1 = x[:-2]
    ...     xm_p1 = x[2:]
    ...     der = np.zeros_like(x)
    ...     der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    ...     der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    ...     der[-1] = 200*(x[-1]-x[-2]**2)
    ...     retour der

Deze gradient informatie is gespecificeerd in de :func:`minimalis` functie
via de ``jac`` parameter zoals hieronder geïllustreerd.


    >>> res = minimaliseren(rosen, x0, method='BFGS', jac=rosen_der,
    ...                opties={'disp': True})
    Optimalisatie succesvol beëindigd.
             Huidige functie waarde: 0.000000
             Iteraties: 51
             Functiebeoordelingen: 63
             Beoordelingen: 63
    >>> print(res.x)
    [ 1. 1. 1. 1. 1. 1.]


Newton-Conjugate-Gradient algoritme (``method='Newton-CG'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^

De methode die de minste functies vereist, is dus vaak
de snelste methode om functies van veel variabelen te minimaliseren gebruikt de
Newton-Conjugate Gradient algoritme. Deze methode is een gewijzigd Newton's
methode en gebruik een conjugate gradient algoritme naar (ongeveer) invert
de lokale Hessian.  De Newton methode is gebaseerd op het passend maken van de functie
lokaal tot een kwadratisch formulier:

.. wiskunde::
   :nowrap:

    \[ f\left(\mathbf{x}\right)\approx f\left(\mathbf{x}_{0}\right)+\nabla f\left(\mathbf{x}_{0}\right)\cdot\left(\mathbf{x}-\mathbf{x}_{0}\right)+\frac{1}{2}\left(\mathbf{x}-\mathbf{x}_{0}\right)^{T}\mathbf{H}\left(\mathbf{x}_{0}\right)\left(\mathbf{x}-\mathbf{x}_{0}\right).\]

waar :math:`\wiskunde{H}\left(\mathbf{x}_{0}\right)` is een matrix van tweede-derivaten (de Hessian). Als de Hessian is
positief, dan kan het lokale minimum van deze functie worden gevonden
door de graduatie van het kwadratische formulier op nul te zetten, resulterend in

.. wiskunde::
   :nowrap:

    \[ \mathbf{x}_{\textrm{opt}}=\mathbf{x}_{0}-\wiskunde{H}^{-1}\nabla f.\]

De omgekeerd van de Hessian wordt geëvalueerd met behulp van de conjugate-gradient
methode. Een voorbeeld van het gebruik van deze methode om de te minimaliseren
Rosenbrock functie wordt hieronder gegeven. Om volledig te profiteren van de
Newton-CG methode, een functie die de Hessian berekent moet zijn
gegeven. De Hessian matrix zelf hoeft niet te worden gebouwd,
alleen een vector die het product is van de Hessian met een willekeurige
vector moet beschikbaar zijn voor de minimization routine. Als gevolg hiervan,
de gebruiker kan een functie opgeven om de Hessian matrix te berekenen,
of een functie om het product van de Hessian te berekenen met een willekeurige
vector.


Volledig Hessian voorbeeld:
"""""""""""""""""""""""""""""

De Hessian van de Rosenbrock functie is

.. wiskunde::
   :nowrap:

    \begin{eqnarray*} H_{ij}=\frac{\partial^{2}f}{\partial x_{i}\partial x_{j}} & = & 200\left(\delta_{i,j}-2x_{i-1}\delta_{i-1,j}\right)-400x_{i}\left(\delta_{i+1,j}-2x_{i}\delta_{i,j}\right)-400\delta_{i,j}\left(x_{i+1}-x_{i}^{2}\right)+2\delta_{i,j},\\  & = & \left(202+1200x_{i}^{2}-400x_{i+1}\right)\delta_{i,j}-400x_{i}\delta_{i+1,j}-400x_{i-1}\delta_{i-1,j},\end{eqnarray*}

als :math:`i,j\in\left[1,N-2\right]` met :math:`i,j\in\left[0,N-1\right]` de :math:`N\times N` matrix definieert. Andere niet-nulentries van de matrix zijn

.. wiskunde::
   :nowrap:

    \begin{eqnarray*} \frac{\partial^{2}f}{\partial x_{0}^{2}} & = & 1200x_{0}^{2}-400x_{1}+2,\\ \frac{\partial^{2}f}{\partial x_{0}\partial x_{1}}=\frac{\partial^{2}f}{\partial x_{1}\partial x_{0}} & = & -400x_{0},\\ \frac{\partial^{2}f}{\partial x_{N-1}\partial x_{N-2}}=\frac{\partial^{2}f}{\partial x_{N-2}\partial x_{N-1}} & = & -400x_{N-2},\\ \frac{\partial^{2}f}{\partial x_{N-1}^{2}} & = & 200.\end{eqnarray*}

Zo is de Hessian wanneer :math:`N=5` is

.. wiskunde::
   :nowrap:

    \[ \mathbf{H}=\left[\begin{array}{ccccc} 1200x_{0}^{2}-400x_{1}+2 & -400x_{0} & 0 & 0 & 0\\ -400x_{0} & 202+1200x_{1}^{2}-400x_{2} & -400x_{1} & 0 & 0\\ 0 & -400x_{1} & 202+1200x_{2}^{2}-400x_{3} & -400x_{2} & 0\\ 0 &  & -400x_{2} & 202+1200x_{3}^{2}-400x_{4} & -400x_{3}\\ 0 & 0 & 0 & -400x_{3} & 200\end{array}\right].\]

De code die deze Hessian berekent, samen met de code om te minimaliseren
de functie met behulp van Newton-CG methode wordt getoond in het volgende voorbeeld:

    >>> rozen_hess(x) ontdooien:
    ...     x = np.asarray(x)
    ...     H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    ...     diagonaal = np.zeros_zoals(x)
    ...     diagonaal[0] = 1200*x[0]**2-400*x[1]+2
    ...     diagonaal[-1] = 200
    ...     diagonaal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    ...     H = H + np.diag(diagonaal)
    ...     retour H

    >>> res = minimaliseren(rosen, x0, method='Newton-CG',
    ...                jac=rosen_der, hess=rosen_hess,
    ...                opties={'avextol': 1e-8, 'disp': True})
    Optimalisatie succesvol beëindigd.
             Huidige functie waarde: 0.000000
             Iteraties: 19
             Functiebeoordelingen: 22
             Beoordelingen: 19
             Hessian evaluaties: 19
    >>> print(res.x)
    [ 1. 1. 1. 1. 1. 1.]


Hessian product voorbeeld:
""""""""""""""""""""""""""""""""""""

Voor grotere minimiatieproblemen, kan de hele Hessian matrix opgeslagen worden
verbruikt aanzienlijke tijd en geheugen. Het Newton-CG algoritme heeft alleen behoefte
het product van de Hessian is soms een willekeurige vector. Als gevolg hiervan, de gebruiker
kan code invoeren om dit product te berekenen in plaats van de volledige Hessian door
een ``hess`` functie geven die de minimization vector als eerste neemt
argument en willekeurige vector als tweede argument (naast extra argument
argumenten die naar de functie zijn doorgestuurd om te worden geminimaliseerd). Indien mogelijk, gebruik
Newton-CG met de optie Hessian product is waarschijnlijk de snelste manier om
de functie minimaliseren.

In dit geval is het product van de Rosenbrock Hessian met een willekeurige willekeurige
vector is niet moeilijk te berekenen. Als :math:`\wiskunde{p}` de willekeurige is
vector, dan :math:`\wiskunde{H}\left(\wiskunde{x}\right)\wiskunde{p}` heeft
elementen:

.. wiskunde::
   :nowrap:

    \[ \mathbf{H}\left(\mathbf{x}\right)\mathbf{p}=\left[\begin{array}{c} \left(1200x_{0}^{2}-400x_{1}+2\right)p_{0}-400x_{0}p_{1}\\ \vdots\\ -400x_{i-1}p_{i-1}+\left(202+1200x_{i}^{2}-400x_{i+1}\right)p_{i}-400x_{i}p_{i+1}\\ \vdots\\ -400x_{N-2}p_{N-2}+200p_{N-1}\end{array}\right].\]

Code die gebruik maakt van dit Hessian product om de te minimaliseren
Rosenbrock functie met :func:`minimalis` volgt:

    >>> ontdek rosen_hess_p(x,p):
    ...     x = np.asarray(x)
    ...     Hp = np.zeros_zoals(x)
    ...     Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
    ...     Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
    ...                -400*x[1:-1]*p[2:]
    ...     Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
    ...     retour Hp

    >>> res = minimaliseren(rosen, x0, method='Newton-CG',
    ...                jac=rosen_der, hess=rosen_hess_p,
    ...                opties={'avextol': 1e-8, 'disp': True})
    Optimalisatie succesvol beëindigd.
             Huidige functie waarde: 0.000000
             Iteraties: 20
             Functiebeoordelingen: 23
             Beoordelingen: 20
             Hessian evaluaties: 44
    >>> print(res.x)
    [ 1. 1. 1. 1. 1. 1.]


.. _tutorial-sqlsp:

Constructieve minimalisering van multivariate schaalfuncties (:func:`minimalis`)
----------------------------------------------------------------------------

De :func:`minimaliseren` functie biedt ook een interface naar meerdere
ingeperkt minimalisering algoritme. Als voorbeeld noem ik de Sequentiële Minst
SQuares Programmeren optimalisatiealgoritme (SLSQP) zal hier worden overwogen.
Dit algoritme maakt het mogelijk om de beperkte minimiatieproblemen van de
formulier:

.. wiskunde::
   :nowrap:

     \begin{eqnarray*} \min F(x) \\ \text{subject to } & C_j(X) =  0   &j = 1,...,\text{MEQ}\\
            & C_j(x) \geq 0  ,  &j = \text{MEQ}+1,...,M\\
           & XL  \leq x \leq XU , &I = 1,...,N. \end{eqnarray*}


Laten we bijvoorbeeld het probleem van de maximalisatie van de functie bekijken:

.. wiskunde::
    :nowrap:

    \[ f(x, y) = 2 x y + 2 x - x ^2 - 2 y ^2 \]

onderworpen aan een gelijkheid en een ongelijkheid beperking die gedefinieerd is als:

.. wiskunde::
    :nowrap:

    \[ x ^3 - y = 0 \]
    \[ y - 1 \geq 0 \]



De objectieve functie en de afgeleide functie daarvan zijn als volgt gedefinieerd.

    >>> def func(x, sign=1.0):
    ...     """" Doelfunctie """
    ...     retourneer teken*(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)

    >>> def func_deriv(x, sign=1.0):
    ...     """ Derivatief van objectieve functie """
    ...     dfdx0 = teken*(-2*x[0] + 2*x[1] + 2)
    ...     dfdx1 = teken*(2*x[0] - 4*x[1])
    ...     retourneer np.array([ dfdx0, dfdx1 ])

Let op dat sinds :func:`minimaliseren` alleen functies minimaliseert, de ``sign``
parameter wordt ingevoerd om de objectieve functie te vermenigvuldigen (en haar
afgeleid door -1) om een maximisatie uit te voeren.

Vervolgens worden beperkingen gedefinieerd als een reeks woordenboeken, met sleutels
``type``, ``fun`` en ``jac``.

    >>> cons = ({'type': 'eq',
    ...          'leuk' : lambda x: np.array([x[0]**3 - x[1]]),
    ...          'jac' : lambda x: np.array([3.0*(x[0]**2.0), -1.0]},
    ...         {'type': 'ineq',
    ...          'leuk' : lambda x: np.array([x[1] - 1]),
    ...          'jac' : lambda x: np.array([0.0, 1.0])})


Nu kan een onbegonnen optimalisatie worden uitgevoerd als:

    >>> res = minimaliseren(func, [-1.0,1.0], args=(-1.0,), jac=func_deriv,
    ...                methode='SLSQP', opties={'disp': True})
    Optimalisatie succesvol beëindigd.    (Verlaat modus 0)
                Huidige functie waarde: -2.0
                Iteraties: 4
                Functiebeoordelingen: 5
                Beoordelingen: 4
    >>> print(res.x)
    [ 2. 1.]

en een beperkte optimalisatie als:

    >>> res = minimaliseren(func, [-1.0,1.0], args=(-1.0,), jac=func_deriv,
    ...                beperkingen=cons, methode='SLSQP', opties={'disp': True})
    Optimalisatie succesvol beëindigd.    (Verlaat modus 0)
                Huidige functie waarde: -1.00018311
                Iteraties: 9
                Functiebeoordelingen: 14
                Beoordelingen: 9
    >>> print(res.x)
    [ 1.00009  1.        ]


Minst-vierkante uitrusting (:func:`leastsq`)
----------------------------

Alle eerder uitgelegd minimiatieprocedures kunnen worden gebruikt om te worden
een probleem op de minst-vierkanten oplossen, mits het juiste doel is
functie is gebouwd. Bijvoorbeeld, denkt u dat het gewenst is om een passend systeem te gebruiken
set of data :math:`\left\{\mathbf{x}_{i}, \wiskunde{y}_{i}\right\}`
naar een bekend model,
:math:`\wiskunde{y}=\wiskunde{f}\left(\wiskunde{x},\wiskunde{p}\right)`
waar :math:`\mathbf{p}` een vector van parameters is voor het model dat dit is
moet worden gevonden. Een gemeenschappelijke methode om te bepalen welke parameter
vector geeft het beste aan de gegevens door de som van vierkanten te minimaliseren
van de residuanen. Het resterende is meestal gedefinieerd voor elke waargenomen
data-punt als

.. wiskunde::
   :nowrap:

    \[ e_{i}\left(\mathbf{p},\mathbf{y}_{i},\mathbf{x}_{i}\right)=\left\Vert \mathbf{y}_{i}-\wiskunde{f}\left(\mathbf{x}_{i},\wiskunde {i}{p}\right)\right\Vert .\]

Een objectieve functie om over te gaan naar een van de vorige minization
algoritmen om een minimum-vierkante fit te krijgen zijn dat.

.. wiskunde::
   :nowrap:

    \[ J\left(\mathbf{p}\right)=\sum_{i=0} ^{N-1}e_{i}^{2}\left(\mathbf{p}\right)\]



Het :obj:`leastsq` algoritme voert deze viering en samenvatting van de
resuals automatisch. Het neemt als invoerargument de vector in beslag
functie :math:`\wiskunde{e}\left(\wiskunde{p}\right)` en retourneert de
waarde van :math:`\wiskunde{p}` welke minimaliseert
:math:`J\left(\wiskunde{p}\right)=\wiskunde{e}^{T}\wiskunde{e}`
direct. De gebruiker wordt ook aangemoedigd om de Jacobiaanse matrix te geven
van de functie (met derivaten omlaag de kolommen of over de hele linie
rijen). Als de Jacobian niet wordt geleverd, wordt het geschat.

Een voorbeeld moet het gebruik verduidelijken. Ondersteund het wordt geloofd dat er sommigen zijn
gemeten gegevens volgen een sinusoidal patroon

.. wiskunde::
   :nowrap:

    \[ y_{i}=A\sin\left(2\pi kx_{i}+\theta\right)\]

waar de parameters :math:`A,` :math:`k` , en :math:`\theta` onbekend zijn. De resterende vector is

.. wiskunde::
   :nowrap:

    \[ e_{i}=\left|y_{i}-A\sin\left(2\pi kx_{i}+\theta\right)\right|.\]

Door een functie te definiëren om de resuanen te berekenen en (selecteer een
passende startpositie), de minst-vierkanten passen routine kan zijn
gebruikt om de beste-fit parameters te vinden :math:`\hat{A},\,\hat{k},\,\hat{\theta}`.
Dit wordt getoond in het volgende voorbeeld:

.. plot::

   >>> van numpy import *
   >>> x = arange(0,6e-2,6e-2/30)
   >> A,k,theta = 10, 1.0/3e-2, pi/6
   >>> y_true = A*sin(2*pi*k*x+theta)
   >>> y_meas = y_true + 2*willekeurig.randn(len(x))

   >>> resuals (p, y, x) ongedaan maken:
   ...     A,k,theta = p
   ...     err = y-A*sin(2*pi*k*x+theta)
   ...     terugzendingerr

   >>> val(x, p) def. peval(x):
   ...     retour p[0]*sin(2*pi*p[1]*x+p[2])

   >> p0 = [8, 1/2.3e-2, pi/3]
   >>> print(array(p0))
   [  8.      43.4783   1.0472]

   >>> van scipy.geoptimaliseerd import leastsq
   >> plsq = leastsq(residuals, p0, args=(y_meas, x))
   >>> print(plsq[0])
   [ 10.9437  33.3605   0.5834]

   >>> print(array([A, k, theta]))
   [ 10.      33.3333   0.5236]

   >>> importeer matplotlib.pyplot als plt
   >>> plt.plot(x,peval(x,plsq[0]),x,y_meas,'o,x,y_true)
   >>> plt.title('Minst-squares passen bij lawisy data')
   >>> plt.legend(['Fit', 'Noisy', 'Juist'])
   >>> plt.show()

..   :caption: Minst-af-van-montage voor lawaaierige gegevens met behulp van
..             :obj:`scipy.optimize.leastsq`


Functie minimaliseren (:func:`minimaliseren_scalar`)
--------------------------------------------------------

Vaak is het minimum van een unieke functie (dat wil zeggen. een functie die
neemt een schaalver als input) is nodig. Onder deze omstandigheden, andere
Optimalisatietechnieken zijn ontwikkeld die sneller kunnen werken. Dit zijn
toegankelijk voor de :func:`minimaliseren_scalar` functie die meerdere voorstelt
algoritmen.


Onbeperkt minimalisering (``method='brent'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^ ^ ^ ^ ^

Er zijn eigenlijk twee methoden die kunnen worden gebruikt om een univarial te minimaliseren
functie: `brent` en `golden`, maar `golden` is alleen opgenomen voor academische
doeleinden en zelden mogen worden gebruikt. Deze kunnen respectievelijk worden geselecteerd
via de `method` parameter in :func:`minimaliseren_scalar`. De 'brent`
methode gebruikt het algoritme van Brent om een minimum te lokaliseren. Optimalig een zakje
(de parameter 'bs' moet worden gegeven die het minimum waar gewenst is bevat. A
haakje is een drievoudige :math:`\left( a, b, c \right)` zo'n :math:`f
\left( een \right) > f \left( b \right) < f \left( c \right)` en :math:`a <
b < c` . Als dit niet wordt gegeven, kunnen er twee startpunten zijn.
wordt gekozen en er wordt een haakje gevonden uit deze punten met een eenvoudige
mars algoritme. Als deze twee startpunten niet worden opgegeven `0` en
`1` zal worden gebruikt (dit is misschien niet de juiste keuze voor uw functie en
resulteert in een onverwacht minimum dat wordt teruggestuurd).

Hier is een voorbeeld:

    >>> van scipy.geoptimaliseerd import minimaliseren_scalar
    >>> f = lambda x: (x - 2) * (x + 1)**2
    >>> res = minimaliseren_scalar(f, method='brent')
    >>> print(res.x)
    1.0


Gecreëerde minimisatie (``method='grens'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^ ^^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

Heel vaak zijn er beperkingen die op de oplossing ruimte kunnen worden geplaatst
voordat de minimalisering plaatsvindt. De 'begrenze' methode in :func:'minimaliseren_scalar'
is een voorbeeld van een beperkte minimiatieprocedure die een
rudimentaire interval beperking voor schaduwfuncties. De interval
Beperkt maakt het mogelijk dat de minimisatie slechts tussen twee vaste
eindpunten, gespecificeerd met behulp van de verplichte 'bs' parameter.

Bijvoorbeeld om het minimum van :math:`J_{1}\left( x \right)` in de buurt te vinden
:math:`x=5` , :func:`minimize_scalar` kan worden aangeroepen met behulp van de interval
:math:`\left[ 4, 7 \right]` als een beperking. Het resultaat is
:math:`x_{\textrm{min}}=5.3314` :

    >>> van scipy.speciale import j1
    >>> res = minimaliseren_scalar(j1, bs=(4, 7), methode='begrenzen')
    >>> print(res.x)
    5.33144184241


Root vinden
------------

Scalar functies
^^^^^^^^^^^^^^^^^^^^^^ ^ ^

Als je een enkele variabele vergelijking hebt, zijn er vier verschillende root
algoritmen vinden die kunnen worden geprobeerd. Elk van deze algoritmen vereist de
eindpunten van een interval waarin een root wordt verwacht (omdat de functie is
wijzigingen tekenen). In het algemeen :obj:`brentq` is de beste keuze, maar de andere
methoden kunnen in bepaalde omstandigheden of voor academische doeleinden nuttig zijn.


Vast-punt oplossen
^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^ ^ ^

Een probleem dat nauw verband houdt met het vinden van de zeros van een functie is de
probleem met het vinden van een vaste functie. Een vast punt van een
functie is het punt waarop de beoordeling van de functie terugkeert
punt: :math:`g\left(x\right)=x.` Het vaste punt van :math:`g`
is de hoofdmap van :math:`f\left(x\right)=g\left(x\right)-x.`
Equivalig, de root van :math:`f` is het vaste_punt van
:math:`g\left(x\right)=f\left(x\right)+x.` De routine
:obj:`fixed_point` biedt een eenvoudige iteratieve methode met Aitkens
sequentie versnelling om het vaste punt van :math:`g` te schatten
startpunt.

Stellen van vergelijkingen
^^^^^^^^^^^^^^^^^^^^^^^^ ^ ^

Een root van een set niet-lineaire vergelijkingen vinden kan worden bereikt met behulp van de
:func:`root` functie. Verschillende methoden zijn beschikbaar, waaronder ``hybr``
(de standaard) en ``lm`` die respectievelijk de hybride methode van Powell gebruiken
en de Levenberg-Marquardt-methode van MINPACK.

Het volgende voorbeeld houdt verband met de enkele variabele transcendental
vergelijking

.. wiskunde::
   :nowrap:

    \[ x+2\cos\left(x\right)=0,\]

een wortel waarvan de volgende gevonden kan worden:

    >>> Importeer numpy als np
    >>> van scipy.geoptimaliseerd import root
    >>> def func(x):
    ...     retour x + 2 * np.cos(x)
    >>> sol = root(func, 0.3)
    >>> sol.x
    array ([-1.02986653])
    >>> sol.fun
    array([ -6.66133815e-16])

Overweeg nu een set niet-lineaire vergelijkingen

.. wiskunde::
   :nowrap:

    \begin{eqnarray*}
    x_{0}\cos\left(x_{1}\right) & = & 4,\\
    x_{0}x_{1}-x_{1} & = & 5.
    \end{eqnarray*}

We definiëren de objectieve functie, zodat deze ook de Jacobiaanse en
geef dit aan door de ``jac`` parameter aan ``True``. Ook de
Levenberg-Marquardt solver wordt hier gebruikt.

::

    >>> def func2(x):
    ...     f = [x[0] * np.cos(x[1]) - 4,
    ...          x[1]*x[0] - x[1] - 5]
    ...     df = np.array([[np.cos(x[1]), -x[0] * np.sin(x[1]),
    ...                    [x[1], x[0] - 1]])
    ...     retour f, df
    >>> sol = root(func2, [1, 1], jac=True, method='lm')
    >>> sol.x
    array([ 6.50409711,  0.90841421])


Root zoeken voor grote problemen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^ ^ ^ ^ ^

Methoden ``hybr`` en ``lm`` in :func:`root` kunnen niet met een zeer groot worden behandeld
aantal variabelen (*N*), omdat ze een dichte *N moeten berekenen en invult
x N* Jacobiaanse matrix op elke Newton stap. Dit wordt nogal inefficiënt
wanneer *N* groeit.

Neem bijvoorbeeld het volgende probleem: we moeten de oplossing vinden
volgen integrodifferentiële vergelijking op het vierkant
:math:`[0,1]\times[0,1]`:

.. wiskunde::

   (\deel_ x^2 + \deel_ y^2) P + 5 \left(\int_0^1\int_0^1\cosh(P)\,dx\,dy\right)^2 = 0

met de grens voorwaarde :math:`P(x,1) = 1` aan de bovenkant en
:math:`P=0` elders op de grens van het vierkant. Dit kan worden gedaan
door de continue functie *P* te benaderen met de waarden op een raster,
:math:`P_{n,m}\ca{}P(n h, m h)`, met een kleine rasterwerving
*u*. De derivaten en integrals kunnen dan worden geharmoniseerd; voor
instance :math:`\deel_ x ^2 P(x,y)\ca{}(P(x+h,y) - 2 P(x,y) +
P(x-h,y))/h ^2`. Het probleem is dan gelijk aan het vinden van de wortel van
een functie ``residual(P)``, waar ``P`` een vector van lengte is
:math:`N_x N_y`.

Nu, omdat :math:`N_x N_y` groot kan zijn, methoden ``hybr`` of ``lm`` in
:func:`root` zal een lange tijd duren om dit probleem op te lossen. De oplossing kan
Maar je vindt een van de grote solanten, bijvoorbeeld
``krylov``, ``broyden2``, of ``anderson``. Dit gebruik wat bekend staat als de
Newton methode niet gebruiken, die in plaats van de Jacobiaanse matrix te berekenen
vormt precies een onderlinge aanpassing.

Het probleem dat we nu hebben, kan als volgt worden opgelost:

.. plot::

    numpy importeren als np
    van scipy.geoptimaliseerd import root
    van numpy import cosh, zeros_like, mgrid, zeros

    # parameters
    nx, ny = 75, 75
    hx, hy = 1./(nx-1), 1./(ny-1)

    P_links, P_rechts = 0, 0
    P_top, P_bottom = 1, 0

    restje leegmaken (P):
       d2x = zeros_like(P)
       d2y = zeros_like(P)

       d2x[1:-1] = (P[2:] - 2*P[1:-1] + P[:-2]) / hx/hx
       d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
       d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

       d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
       d2y[:,0]  = (P[:,1] - 2*P[:,0]  + P_bottom)/hy/hy
       d2y[:,-1]   = (P_top - 2*P[:,-1]   + P[:,-2])/hy/hy

       retourneer d2x + d2y + 5*cosh(P).mean()**2

    # oplossen
    raden = zeros((nx, ny), zwef)
    sol = root(resual, guess, method='krylov', opties={'disp': True})
    #sol = root(resent, raden, methode='broyden2', opties={'disp': True, 'max_rank': 50})
    #sol = root(resual, guess, method='anderson', opties={'disp': True, 'M': 10})
    print('Residual: %g' % abs(resual(sol.x)).max())

    # visualiseren
    importeer matplotlib.pyplot als plt
    x, y = mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
    plt.pcolor(x, y, sol.x)
    plt.colorbar()
    plt.show()


Nog te langzaam? Voorconditioning.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^ ^ ^ ^

Bij het zoeken naar de nul van de functies :math:`f_i({\bf x}) = 0`,
*i = 1, 2, ..., N*, de ``krylov`` solver spendeert het meeste van zijn
tijd omdraaien van de Jacobiaanse matrix,

.. wiskunde:: J_{ij} = \frac{\deelf_i}{\deelx_j} .

Als u een onderlinge aanpassing voor de omgekeerd matrix hebt
:math:`M\ca{}J ^{-1}`, je kunt het gebruiken voor *preconditioning* de
lineaire inversie probleem. Het idee is dat in plaats van op te lossen
:math:`J{\bf s}={\bf y}` one solves :math:`MJ{\bf s}=M{\bf y}`: sinds
matrix :math:`MJ` is "dichter" bij de identiteitsmatrix dan :math:`J`
is dat de vergelijking gemakkelijker moet zijn voor de Krylov-methode.

De matrix *M* kan worden doorgegeven aan :func:`root` met methode ``krylov`` als een
optie ``options['jac_options']['inner_M']``. Het kan een (vonk) matrix zijn
of een :obj:`scipy.sparse.linalg.LinearOperator` instantie.

Voor het probleem in de vorige sectie, stellen we vast dat de functie om
oplossing bestaat uit twee delen: de eerste is toepassing van de
Laplace operator, :math:`[\deel_ x^2 + \partial_y^2] P`, en de tweede
is het integraal. We kunnen de Jacobiaanse corresponderende gemakkelijk berekenen
naar het onderdeel Laplace operator: we weten dat in één dimensie

.. wiskunde::

   \deel_ x ^2 \ca \frac{1}{h_x^2} \begin{pmatrix}
   -2 & 1 & 0 & 0 \cdots \\
   1 & -2 & 1 & 0 \cdots \\
   0 & 1 & -2 & 1 \cdots \\
   \ldots
   \eind{pmatrix}
   = h_x ^{-2} L

zodat de gehele 2-D operator wordt vertegenwoordigd door

.. wiskunde::

   J_1 = \deel_ x + \deelgenoot_y^2
   \simeq
   h_x ^{-2} L \otimes I + h_y ^{-2} I \otimes L

De matrix :math:`J_2` van de Jacobian die overeenkomt met het integraal
is moeilijker te berekenen, en aangezien *alle* van het items zijn
non zero, het zal moeilijk zijn om te invult. :math:`J_1` aan de andere kant
is een relatief eenvoudige matrix, en kan omgekeerd worden door
:obj:`scipy.sparse.linalg.splu` (of de inverse kan worden geschgeschatte door
:obj:`scipy.sparse.linalg.spilu`). Dus we zijn tevreden om te nemen
:math:`M\ca{}J_1 ^{-1}` en hoop voor het beste.

In het onderstaande voorbeeld gebruiken we de preconditioner :math:`M=J_1 ^{-1}`.

.. literalinclude:: voorbeelden/newton_krylov_preconditioning.py

Resultaat loopt, eerst zonder voorafgaande voorwaarden::

  0:  |F(x)| = 803.614; stap 1; tol 0.000257947
  1:  |F(x)| = 345.912; stap 1; tol 0.166755
  2:  |F(x)| = 139.159; stap 1; tol 0.145657
  3:  |F(x)| = 27.3682; stap 1; tol 0.0348109
  4: |F(x)| = 1.03303; stap 1; tol 0.00128227
  5:  |F(x)| = 0.0406634; stap 1; tol 0.00139451
  6: |F(x)| = 0.00344341; stap 1; tol 0.00645373
  7: |F(x)| = 0.000153671; stap 1; tol 0.00179246
  8:  |F(x)| = 6.7424e-06; stap 1; tol 0.00173256
  Resterende 3.57078908664e-07
  Evaluaties 317

en dan met preconditionering::

  0:  |F(x)| = 136.993; stap 1; tol 7.49599e-06
  1:  |F(x)| = 4.80983; stap 1; tol 0.00110945
  2: |F(x)| = 0.195942; stap 1; tol 0.00149362
  3:  |F(x)| = 0.000563597; stap 1; tol 7.44604e-06
  4:  |F(x)| = 1.00698e-09; stap 1; tol 2.87308e-12
  Resterende 9.29603061195e-11
  Evaluaties 77

Door een preconditioner te gebruiken is het aantal beoordelingen van de
``residual`` functie door een factor van *4*. Voor problemen waar de
restje is duur om te berekenen, goede precitionering kan cruciaal zijn
--- Het kan zelfs beslissen of het probleem in de praktijk oplosbaar is of
niet.

Preconditioning is een kunst, wetenschap en industrie. Hier, we hadden geluk
door een eenvoudige keuze te maken die redelijk goed werkte, maar er is een
veel diepgaander aan dit onderwerp dan hier wordt getoond.

.. rubric:: Referenties

Een aantal andere lezingen en gerelateerde software voor het oplossen van grootschalige problemen
( [PP]_, [AMG]_):

.. [KK] D.A. Knoll en D.E. Sleutels "Jacobian-free Newton-Krylov methoden",
        J. Comp. Fooien. 193, 357 (2003).

.. [PP] PETSc http://www.mcs.anl.gov/petsc/ en zijn Python bindingen
        http://code.google.com/p/petsc4py/

.. [AMG] PyAMG (algebraïsche multiraster preconditioners/solvers)
         http://code.google.com/p/pyamg/
