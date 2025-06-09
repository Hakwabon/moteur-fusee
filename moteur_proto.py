import numpy as np
import matplotlib.pyplot as plt
import thermochem.burcat as thermo
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import ConvexHull
import matplotlib.animation as animation
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Moteur Fusée")
        self.geometry("1000x700")

        self.use_fixed_temp = tk.BooleanVar()
        self.temp_combustion = tk.StringVar(value="2000")
        self.richesse = tk.StringVar(value="1")
        self.Pression = tk.StringVar(value="4")
        self.debit = tk.StringVar(value="10")

        self.carburant_selection = tk.StringVar()
        self.carburant_predefinis = ["essence", "CH4", "H2"]
        self.carburant_selection.set(self.carburant_predefinis[0])

        self.calcul_selection = tk.StringVar()
        self.calcul_predefinis = ["forme", "valeurs 2D", "valeurs 3D", "Thermique"]
        self.calcul_selection.set(self.calcul_predefinis[0])

        # ====================
        # FRAME PRINCIPAL HAUT
        # ====================
        top_frame = tk.Frame(self)
        top_frame.grid(row=0, column=0, sticky="nsew")

        # ====================
        # ZONE BOUTONS GAUCHE
        # ====================
        controls_frame = tk.Frame(top_frame)
        controls_frame.grid(row=0, column=0, sticky="nw", padx=10, pady=8)

        # Carburant
        tk.Label(controls_frame, text="Carburant").pack()
        tk.OptionMenu(controls_frame, self.carburant_selection, *self.carburant_predefinis).pack(pady=5)

        # Richesse
        richesse_frame = tk.Frame(controls_frame)
        richesse_frame.pack(pady=5)
        tk.Label(richesse_frame, text="Richesse:").pack(side=tk.LEFT)
        tk.Entry(richesse_frame, textvariable=self.richesse, width=8).pack(side=tk.LEFT)

        # Température
        temp_frame = tk.Frame(controls_frame)
        temp_frame.pack(pady=5)
        tk.Checkbutton(temp_frame, text="Température imposée", variable=self.use_fixed_temp).pack(side=tk.LEFT)
        tk.Entry(temp_frame, textvariable=self.temp_combustion, width=8).pack(side=tk.LEFT)
        tk.Label(temp_frame, text="K").pack(side=tk.LEFT)

        # Pression
        pressure_frame = tk.Frame(controls_frame)
        pressure_frame.pack(pady=5)
        tk.Label(pressure_frame, text="Pression:").pack(side=tk.LEFT)
        tk.Entry(pressure_frame, textvariable=self.Pression, width=8).pack(side=tk.LEFT)
        tk.Label(pressure_frame, text="bar").pack(side=tk.LEFT)

        # Débit
        debit_frame = tk.Frame(controls_frame)
        debit_frame.pack(pady=5)
        tk.Label(debit_frame, text="Débit:").pack(side=tk.LEFT)
        tk.Entry(debit_frame, textvariable=self.debit, width=8).pack(side=tk.LEFT)
        tk.Label(debit_frame, text="g/s").pack(side=tk.LEFT)

        # Lancer
        tk.Button(controls_frame, text="Lancer simulation moteur", command=self.run_simulation).pack(pady=10)

        # Type de calcul
        tk.Label(controls_frame, text="Graphique").pack()
        tk.OptionMenu(controls_frame, self.calcul_selection, *self.calcul_predefinis).pack(pady=5)

        tk.Button(controls_frame, text="affichage graphique", command=self.show_graph).pack(pady=10)

        # ====================
        # ZONE TEXTE À DROITE
        # ====================
        self.text = tk.Text(top_frame, wrap="word", width=40,height=6)
        self.text.grid(row=0, column=1, sticky="nsew", padx=10, pady=12)

        # ====================
        # ZONE GRAPHIQUE EN BAS
        # ====================
        self.graph_frame = tk.Frame(self)
        self.graph_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=15)

        # Expansion layout
        self.grid_rowconfigure(0, weight=0)  # Texte + boutons : prennent peu de place
        self.grid_rowconfigure(1, weight=3)  # Graphique : prend le plus de place

        self.grid_columnconfigure(0, weight=1)  # Pour que tout s'étale bien
        top_frame.grid_columnconfigure(1, weight=1)
        self.graph_frame.grid_rowconfigure(0, weight=1)
        self.graph_frame.grid_columnconfigure(0, weight=1)

        self.protocol("WM_DELETE_WINDOW", self.quit_application)


    def quit_application(self):
        plt.close('all')
        self.destroy()

    def run_simulation(self):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        self.text.delete("1.0", tk.END)

        phi = float(self.richesse.get())

        reaction = chimie(self.carburant_selection.get(),True, phi)
        pptchimie, cp, r, gamma, RMC = reaction.proprietes()
        # self.text.insert(tk.END, f"RMC :{RMC:.2}\n")
        
        if self.use_fixed_temp.get():
            T_comb = float(self.temp_combustion.get())
        else:
            T_comb = Combustion(pptchimie).Tf

        self.moteur = dimensionnement_moteur(float(self.debit.get())/1000, float(self.Pression.get()), T_comb, gamma,r,7)

        v0 = self.moteur.M0*np.sqrt(gamma*r*T_comb)
        vc = np.sqrt(gamma*r*T_comb*self.moteur.rapport_temperature(1))
        ve = self.moteur.Mmoteur[-1]*np.sqrt(gamma*r*self.moteur.Tmoteur[-1])

        # Instanciation des objets Thermique
        thermique_chambre = Thermique(2*self.moteur.forme.r_chambre, v0)
        thermique_col = Thermique(2*self.moteur.forme.Rc, vc)
        thermique_tuyere = Thermique(2*self.moteur.forme.Re, ve)

        # Lancer les calculs de température (sans afficher directement)
        self.T_chambre, self.X_chambre, self.T_evolution_chambre, self.temps = thermique_chambre.evolution_tempo_temperature(N=100, return_data=True)
        self.T_col, self.X_col, self.T_evolution_col, _ = thermique_col.evolution_tempo_temperature(N=100, return_data=True)
        self.T_tuyere, self.X_tuyere, self.T_evolution_tuyere, _ = thermique_tuyere.evolution_tempo_temperature(N=100, return_data=True)

        # Affichage texte
        self.text.insert(tk.END,f"Propriétés du gaz brulé dans la chambre:\n\
    Température de flamme: {T_comb:.0f} K\n\
    RMC :{RMC:.2}\n\
    gamma :{gamma:.2f}\n\
    vitesse :{v0:.0f}  m/s\n\
    mach : {self.moteur.M0:.4f}\n\
    coefficient thermique h :{thermique_chambre.h_int:.0f} W/m²/K\n\
Propriétés du gaz brulé au col:\n\
    Température : {T_comb*self.moteur.rapport_temperature(1):.0f} K\n\
    gamma :{gamma:.2f}\n\
    vitesse :{vc:.0f} m/s\n\
    mach : 1\n\
    coefficient thermique h :{thermique_col.h_int:.0f} W/m²/K\n\
Propriétés du gaz brulé à la sortie:\n\
    Température : {self.moteur.Tmoteur[-1]:.0f} K\n\
    gamma :{gamma:.2f}\n\
    vitesse :{ve:.0f}  m/s\n\
    mach : {self.moteur.Mmoteur[-1]:.0f}\n\
    coefficient thermique h :{thermique_tuyere.h_int:.0f} W/m²/K\n")
        self.text.insert(tk.END,    f"Dimension du moteur:\n \
Rayon chambre = {self.moteur.forme.r_chambre*100:.1f} cm, Longueur chambre = {self.moteur.forme.L_chambre*100:.1f} cm,\n \
Rayon au col = {self.moteur.forme.Rc*100:.1f} cm, Longueur convergeant = {self.moteur.forme.L_conv*100:.1f} cm, \n \
Rayon en sortie = {self.moteur.forme.Re*100:.1f} cm,Longueur tuyere = {(self.moteur.forme.L_tuyere[-1]-self.moteur.forme.L_chambre-self.moteur.forme.L_conv)*100:.1f} cm")

    def show_graph(self):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        calcul = self.calcul_selection.get()

        if calcul == self.calcul_predefinis[0]:
            fig = self.moteur.forme.affichage_forme_generale(affichage=False)

        elif calcul == self.calcul_predefinis[1]:
            fig = self.moteur.affichage_2D(False)
        
        elif calcul == self.calcul_predefinis[2]:
            fig = self.moteur.affichage_3D(N_theta=30)
        
        elif calcul == self.calcul_predefinis[3]:
            # Création de la figure avec 3 sous-graphes
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # --- Préparation des graphes ---
            lines = []
            titles = ['Chambre', 'Col', 'Tuyère']
            Xs = [self.X_chambre, self.X_col, self.X_tuyere]
            Tevos = [self.T_evolution_chambre, self.T_evolution_col, self.T_evolution_tuyere]

            for ax, X, T_init, title in zip(axs, Xs, Tevos, titles):
                line, = ax.plot(X*1000, T_init[0], lw=2)
                ax.set_xlim(0, max(X)*1000)
                ax.set_ylim(min(T_init[0]) - 50, np.max(Tevos) + 100)
                ax.set_xlabel("Épaisseur de la paroie [mm]")
                ax.set_ylabel("Température [K]")
                ax.set_title(title)
                lines.append(line)

            # --- Ajout de la barre de progression ---
            progress_ax = fig.add_axes([0.1, 0.02, 0.8, 0.02])  # position [gauche, bas, largeur, hauteur]
            progress_bar, = progress_ax.plot([], [], color='limegreen', lw=6)
            progress_ax.set_xlim(0, 1)
            progress_ax.set_ylim(0, 1)
            progress_ax.axis('off')  # pas de graduation, juste la barre verte

            fig.suptitle(f"Évolution des températures - t = 0.00 s")

            def update(frame):
                for line, Tevo in zip(lines, Tevos):
                    line.set_ydata(Tevo[frame])
                fig.suptitle(f"Évolution des températures - t = {self.temps[frame]:.2f} s")

                # Mise à jour de la barre de progression
                progress_bar.set_data([0, frame / (len(self.temps) - 1)], [0.5, 0.5])

                return lines + [progress_bar]

            ani = animation.FuncAnimation(fig, update, frames=len(self.temps), blit=False, interval=10, repeat=False)

        # Affichage graphique
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


class chimie:
    def __init__(self,reaction = "essence", air = True, phi=1):
        self.R = 8.314
        
        if air :
            self.alpha = 3.76
        else :
            self.alpha = 0
        
        if reaction == "essence":
            db = thermo.Elementdb()
            # print(db.search("C8H18"))
            # Définition des espèces avec le bon nom dans la base thermo
            essence = db.getelementdata("C8H18,isooctane")
            O2 = db.getelementdata("O2 REF ELEMENT")
            N2 = db.getelementdata("N2  REF ELEMENT")
            H2O = db.getelementdata("H2O")
            CO2 = db.getelementdata("CO2")

            # Paramètres
            # phi = 1                # Richesse (modifiable)
            # alpha = 3.76             # Rapport molaire N2/O2 dans l'air
            RMC_stoech = 12.5 * O2.mm / essence.mm  # Rapport massique stœchiométrique

            # Ajustement en fonction de la richesse
            RMC = RMC_stoech / phi
            self.RMC = RMC
            m_fuel, m_O2 = self.masse_combu(1, RMC)

            # Calcul du nombre de moles pour chaque espèce
            self.pptchim = {
                "C8H18": {"nom": essence, "mole": m_fuel / essence.mm},
                "O2": {"nom": O2, "mole": m_O2 / O2.mm},
                "N2": {"nom": N2, "mole": self.alpha * m_O2 / O2.mm},
                "CO2": {"nom": CO2, "mole": -8 * m_fuel / essence.mm},
                "H2O": {"nom": H2O, "mole": -9 * m_fuel / essence.mm},
                "N2 produit": {"nom": N2, "mole": -self.alpha * m_O2 / O2.mm}
            }
            self.r = self.R/essence.mm
            self.cp = essence.cp  # J/mol/K
            self.gamma = self.cp / (self.cp - self.r)
        
        elif reaction == "CH4":
            db = thermo.Elementdb()
            # print(db.search("C8H18"))
            CH4 = db.getelementdata("CH4   RRHO")
            N2 = db.getelementdata("N2  REF ELEMENT")
            O2 = db.getelementdata("O2 REF ELEMENT")
            H2O = db.getelementdata("H2O")
            CO2 = db.getelementdata("CO2")
            #butane
            #phi entre 1.8% et 8.4%
            RMC_stoech = 2* O2.mm / CH4.mm  # Rapport massique stœchiométrique

            # Ajustement en fonction de la richesse
            RMC = RMC_stoech / phi
            self.RMC = RMC
            m_fuel, m_O2 = self.masse_combu(1, RMC)

            self.pptchim = { "CH4":{"nom":CH4,"mole":m_fuel/CH4.mm},\
                        "N2":{"nom":N2,"mole":self.alpha*m_O2/O2.mm},\
                        "O2":{"nom":O2,"mole":m_O2/O2.mm},\
                        "H2O":{"nom":H2O,"mole":-2*m_fuel/CH4.mm},\
                        "CO2":{"nom":CO2,"mole":-m_fuel/CH4.mm},
                        "N2 produit":{"nom":N2,"mole":-self.alpha*m_O2/O2.mm}}
            
            # for nom in self.pptchim:
            #     print(nom,self.pptchim[nom]["nom"].hfr*self.R)
            
            self.r = self.R/CH4.mm
            self.cp = CH4.cp  # J/mol/K
            self.gamma = self.cp / (self.cp - self.r)
        
        elif reaction == "H2":
            db = thermo.Elementdb()
            # print(db.search("H2"))
            H2 = db.getelementdata("H2  REF ELEMENT")
            O2 = db.getelementdata("O2 REF ELEMENT")
            N2 = db.getelementdata("N2  REF ELEMENT")
            H2O = db.getelementdata("H2O")

            RMC_stoech = 0.5 * O2.mm / H2.mm  # Rapport massique stœchiométrique

            self.RMC = RMC_stoech / phi
            RMC = self.RMC
            m_fuel, m_O2 = self.masse_combu(1, self.RMC)

            self.pptchim = {
                "H2": {"nom": H2, "mole": m_fuel / H2.mm},
                "O2": {"nom": O2, "mole": m_O2 / O2.mm},
                "N2": {"nom": N2, "mole": self.alpha * m_O2 / O2.mm},
                "H2O": {"nom": H2O, "mole": -m_fuel / H2.mm},
                "N2 produit": {"nom": N2, "mole": -self.alpha * m_O2 / O2.mm}
            }

            self.r = self.R / H2.mm
            self.cp = H2.cp
            self.gamma = self.cp / (self.cp - self.r)

        else :
            print("erreur")
        
    def proprietes(self):
        return self.pptchim, self.cp, self.r, self.gamma, self.RMC
    
    def masse_combu(self,mtot, RMC):
        """
        Calcule les masses de carburant et d'oxygène à partir du rapport massique de combustion (RMC).
        """
        m_fuel = mtot / (1 + RMC)
        m_ox = mtot * RMC / (1 + RMC)
        return m_fuel, m_ox
    
    def T_flamme_phi(self, phi_min=0.5, phi_max=2,nom="CH4", N=100):
        l_phi = np.linspace(phi_min,phi_max,N)
        Tf = []
        for phi in l_phi:
            pptchimie, cp, r, gamma = chimie(nom,True,phi).proprietes()
            Tf.append(Combustion(pptchimie).Tf)
        
        plt.plot(l_phi,Tf)
        plt.xlabel("Richesse")
        plt.ylabel("Température de flamme (K)")
        plt.show()

class Combustion:
    def __init__(self, dico):
        self.dico = dico
        self.R = 8.314  # Constante universelle des gaz parfaits en J/(mol·K)
        self.Tf = self.T_flamme()
        self.E = self.enthalpie_tot(self.Tf)

    def dichotomie(self, valeurs, bsupp=3000, binf=298.15):
        """
        Fonction pour rechercher une température T qui correspond à une valeur donnée d'enthalpie
        en utilisant la méthode de dichotomie.
        Arguments :
        - valeurs : l'enthalpie cible à atteindre.
        - bsupp : borne supérieure pour la recherche de température (par défaut 5000 K).
        - binf : borne inférieure pour la recherche de température (par défaut 298.15 K).
        Retourne :
        - T : température correspondant à l'enthalpie cible, ou la borne la plus proche si elle est hors des limites.
        """
        T = (bsupp + binf) / 2
        E = self.enthalpie(T)

        if valeurs < self.enthalpie_tot(binf) :
            print("Pas de combustion")
            return binf
        
        elif valeurs > self.enthalpie_tot(bsupp):
            print("hors du domaine de validité des équations")
            return bsupp

        while abs(E - valeurs) > 100 and binf < T < bsupp:
            if E < valeurs:
                binf = T
            else:
                bsupp = T

            T = (binf + bsupp) / 2
            E = self.enthalpie(T)

        return T

    def enthalpie_formation(self, T0=298.15):
        E = 0
        for nom in self.dico:
            E += self.dico[nom]["nom"].hfr * (self.dico[nom]["mole"]) * self.R
        return E

    def enthalpie(self, T0=298.15):
        E = 0
        for nom in self.dico:
            if self.dico[nom]["mole"] < 0:
                E -= self.dico[nom]["nom"].ho(T0) * (self.dico[nom]["mole"])
        return E

    def enthalpie_tot(self, T=298.15):
        return self.enthalpie_formation() + self.enthalpie(T)

    def T_flamme(self):
        return self.dichotomie(self.enthalpie_tot(298.15))

class geometrie_old:
    def __init__(self, dmdt, Rc=0.02, Re=0.05, theta=15, S_chambre = np.pi*0.062**2,ratio_chambre=7,ts=4e-3):
        self.Rc = Rc
        self.Re = Re
        self.theta = np.radians(theta)
        self.S_chambre = S_chambre
        # self.S_chambre = ratio_chambre * np.pi * Rc**2
        self.ratio_chambre = ratio_chambre
        self.dmdt = dmdt
        self.theta_conv = np.pi/12
        self.nbr_points = 60
        self.ts = ts # temps de sejour
        self.tuyere()
        self.forme_generale_2D(False)
        
    def chambre(self):
        # t_s = self.ts  # temps de sejour
        # self.L_chambre = t_s * self.dmdt / self.S_chambre
        self.L_chambre = np.sqrt(self.S_chambre/np.pi)*self.ratio_chambre
        self.r_chambre = np.sqrt(self.S_chambre / np.pi)


        # Section chambre
        self.x_chambre = np.linspace(0, self.L_chambre, self.nbr_points)
        self.y_chambre = np.full_like(self.x_chambre, self.r_chambre)

        # Convergent linéaire
        self.L_conv = abs(self.r_chambre - self.Rc) / np.tan(self.theta_conv)
        if self.r_chambre - self.Rc<0:
            print("Erreur : Rc > r_chambre")
        self.x_conv = np.linspace(self.L_chambre, self.L_chambre + self.L_conv, self.nbr_points)
        self.y_conv = self.r_chambre - ((self.r_chambre - self.Rc) / self.L_conv) * (self.x_conv - self.L_chambre)

    def tuyere(self):
        self.chambre()
        x0 = self.L_chambre + self.L_conv  # départ tuyère
        # L_tuy = self.L  # longueur de la partie divergente
        # L_tuy = (self.Re-self.Rc)/np.tan(self.theta)  # longueur de la partie divergente
        L_tuy = 0.7*((self.Re/self.Rc-1)*self.Rc+0.5*self.Rc*np.sin(self.theta))/np.tan(self.theta) # longueur de la partie divergente
 
        def equations(vars):
            a, b, c = vars
            eq1 = c - self.Rc
            eq2 = a * L_tuy**2 + b * L_tuy + c - self.Re
            eq3 = 2 * a * L_tuy + b - np.tan(self.theta)
            return [eq1, eq2, eq3]

        a, b, c = fsolve(equations, (1, 1, 1))

        self.L_tuyere = np.linspace(x0, x0 + L_tuy, self.nbr_points)
        x_local = self.L_tuyere - x0
        self.y_tuyere = a * x_local**2 + b * x_local + c

    def forme_generale_2D(self,affichage=True):
        # print(f"Rc={self.Rc:.4f} m, r_chambre={self.r_chambre:.4f} m, L_chambre={self.L_chambre:.4f} m, L_conv={self.L_conv:.4f} m, L_tuyere={self.L_tuyere[-1]-self.L_chambre-self.L_conv:.4f} m")
        self.x_moteur = np.concatenate((self.x_chambre, self.x_conv, self.L_tuyere))
        self.r_moteur = np.concatenate((self.y_chambre, self.y_conv, self.y_tuyere))

        if affichage:
            plt.plot(self.x_moteur, self.r_moteur,color="grey", label="Forme générale")
            plt.plot(self.x_moteur, -self.r_moteur,color="grey")
            plt.xlabel("Longueur (m)")
            plt.ylabel("Rayon (m)")
            plt.title("Géométrie moteur fusée")
            plt.legend()
            plt.axis("equal")
            plt.show()

    def forme_generale_3D(self, Ntheta=50, affichage=True):
        # Profil du moteur
        self.forme_generale_2D(affichage=False)
        # self.x_moteur = np.concatenate((self.x_chambre, self.x_conv, self.L_tuyere))
        # self.r_moteur = np.concatenate((self.y_chambre, self.y_conv, self.y_tuyere))

        
        # Création de l'angle de révolution (0 à 2pi)
        theta = np.linspace(0, 2 * np.pi, Ntheta)
        self.X, Theta = np.meshgrid(self.x_moteur, theta)
        self.R = np.tile(self.r_moteur, (Ntheta, 1))

        # Coordonnées en 3D
        self.Y = self.R * np.cos(Theta)
        self.Z = self.R * np.sin(Theta)
        if affichage:
            # Tracé 3D
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.X, self.Y, self.Z, alpha=1.0)
            ax.set_xlabel("Longueur (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title("Géométrie 3D du moteur-fusée")
            ax.view_init(elev=20, azim=120)
            plt.tight_layout()
            plt.show()

    def affichage_forme_generale(self, Ntheta=50, affichage=True):
        # Génére les profils
        self.forme_generale_2D(affichage=False)
        self.forme_generale_3D(Ntheta=Ntheta, affichage=False)

        # Création figure avec 2 sous-figures : une 2D et une 3D
        fig = plt.figure(figsize=(10, 8))

        # 1. Affichage 2D
        ax1 = fig.add_subplot(2, 1, 1)  # 2 lignes, 1 colonne, premier subplot
        ax1.plot(self.x_moteur, self.r_moteur, color="grey", label="Forme générale")
        ax1.plot(self.x_moteur, -self.r_moteur, color="grey")
        ax1.set_xlabel("Longueur (m)")
        ax1.set_ylabel("Rayon (m)")
        ax1.set_title("Géométrie moteur fusée (2D)")
        ax1.legend()
        ax1.axis("equal")

        # 2. Affichage 3D
        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        ax2.plot_surface(self.X, self.Y, self.Z, alpha=1.0)
        ax2.set_xlabel("Longueur (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")
        ax2.set_title("Géométrie moteur fusée (3D)")
        ax2.view_init(elev=20, azim=120)

        plt.tight_layout()

        if affichage:
            plt.show()

        return fig

class geometrie:
    def __init__(self, dmdt, Rc=0.02, Re=0.05, theta=15, S_chambre = np.pi*0.062**2,ratio_chambre=7,ts=4e-3,alpha=0):
        self.Rc = Rc
        self.Re = Re
        self.theta = np.radians(theta)
        self.S_chambre = S_chambre
        # self.S_chambre = ratio_chambre * np.pi * Rc**2
        self.ratio_chambre = ratio_chambre
        self.dmdt = dmdt
        self.theta_conv = np.pi/12
        self.nbr_points = 60
        self.ts = ts # temps de sejour
        self.alpha = alpha
        self.col()
        self.forme_generale_2D(False)
        
    def chambre(self):
        # t_s = self.ts  # temps de sejour
        # self.L_chambre = t_s * self.dmdt / self.S_chambre
        self.L_chambre = np.sqrt(self.S_chambre/np.pi)*self.ratio_chambre
        self.r_chambre = np.sqrt(self.S_chambre / np.pi)


        # Section chambre
        self.x_chambre = np.linspace(0, self.L_chambre, self.nbr_points)
        self.y_chambre = np.full_like(self.x_chambre, self.r_chambre)

        # Convergent linéaire
        self.L_conv = abs(self.r_chambre - self.Rc) / np.tan(self.theta_conv)
        if self.r_chambre - self.Rc<0:
            print("Erreur : Rc > r_chambre")
        self.x_conv = np.linspace(self.L_chambre, self.L_chambre + self.L_conv*(1-self.alpha), self.nbr_points)
        self.y_conv = self.r_chambre - ((self.r_chambre - self.Rc) / self.L_conv) * (self.x_conv - self.L_chambre)

    def tuyere(self):
        self.chambre()
        x0 = self.L_chambre + self.L_conv  # départ tuyère
        # L_tuy = self.L  # longueur de la partie divergente
        # L_tuy = (self.Re-self.Rc)/np.tan(self.theta)  # longueur de la partie divergente
        L_tuy = 0.7*((self.Re/self.Rc-1)*self.Rc+0.5*self.Rc*np.sin(self.theta))/np.tan(self.theta) # longueur de la partie divergente
 
        def equations(vars):
            a, b, c = vars
            eq1 = c - self.Rc
            eq2 = a * L_tuy**2 + b * L_tuy + c - self.Re
            eq3 = 2 * a * L_tuy + b - np.tan(self.theta)
            return [eq1, eq2, eq3]

        a, b, c = fsolve(equations, (1, 1, 1))

        self.L_tuyere = np.linspace(x0+self.L_conv*self.alpha, x0 + L_tuy, self.nbr_points)
        x_local = self.L_tuyere - x0
        self.y_tuyere = a * x_local**2 + b * x_local + c

    def col(self):
        self.tuyere()
        if self.alpha == 0:
            self.x_col = np.array([self.L_tuyere[0]])
            self.y_col = np.array([self.Rc])
        else :
            x0 = self.x_conv[-1]
            x1 = self.L_tuyere[0]
            self.x_col =np.linspace(x0,x1,100)
            matrix = np.array([ [x0**5,x0**4,x0**3,x0**2,x0,1],\
                        [x1**5,x1**4,x1**3,x1**2,x1,1],\
                        [5*x0**4,4*x0**3,3*x0**2,2*x0,1,0],\
                        [5*x1**4,4*x1**3,3*x1**2,2*x1,1,0],\
                        [((x0+x1)/2)**5,((x0+x1)/2)**4,((x0+x1)/2)**3,((x0+x1)/2)**2,((x0+x1)/2),1],\
                        [5*((x0+x1)/2)**4,4*((x0+x1)/2)**3,3*((x0+x1)/2)**2,2*((x0+x1)/2),1,0]])

            vec_sol = np.array([[self.y_conv[-1]],\
                                [self.y_tuyere[0]],\
                                [(self.y_conv[-1]-self.y_conv[-2])/(self.x_conv[-1]-self.x_conv[-2])],\
                                [(self.y_tuyere[0]-self.y_tuyere[1])/(self.L_tuyere[0]-self.L_tuyere[1])],\
                                [self.Rc],[0]])
            coef = np.linalg.solve(matrix,vec_sol)
            self.y_col = coef[0]*self.x_col**5+coef[1]*self.x_col**4+coef[2]*self.x_col**3+coef[3]*self.x_col**2+coef[4]*self.x_col+coef[5]


    def forme_generale_2D(self,affichage=True):
        # print(f"Rc={self.Rc:.4f} m, r_chambre={self.r_chambre:.4f} m, L_chambre={self.L_chambre:.4f} m, L_conv={self.L_conv:.4f} m, L_tuyere={self.L_tuyere[-1]-self.L_chambre-self.L_conv:.4f} m")
        if self.alpha == 0:
            self.x_moteur = np.concatenate((self.x_chambre, self.x_conv, self.L_tuyere))
            self.r_moteur = np.concatenate((self.y_chambre, self.y_conv, self.y_tuyere))
        else:
            self.x_moteur = np.concatenate((self.x_chambre, self.x_conv, self.x_col, self.L_tuyere))
            self.r_moteur = np.concatenate((self.y_chambre, self.y_conv, self.y_col, self.y_tuyere))

        if affichage:
            plt.plot(self.x_moteur, self.r_moteur, label=f"Forme générale {self.alpha}")
            plt.plot([self.x_conv[-1],self.L_tuyere[0]],[self.y_conv[-1],self.y_tuyere[0]],"o")
            plt.plot(self.x_moteur, -self.r_moteur,color="grey")
            plt.xlabel("Longueur (m)")
            plt.ylabel("Rayon (m)")
            plt.title("Géométrie moteur fusée")
            plt.legend()
            plt.axis("equal")
            

    def forme_generale_3D(self, Ntheta=50, affichage=True):
        # Profil du moteur
        self.forme_generale_2D(affichage=False)
        # self.x_moteur = np.concatenate((self.x_chambre, self.x_conv, self.L_tuyere))
        # self.r_moteur = np.concatenate((self.y_chambre, self.y_conv, self.y_tuyere))

        
        # Création de l'angle de révolution (0 à 2pi)
        theta = np.linspace(0, 2 * np.pi, Ntheta)
        self.X, Theta = np.meshgrid(self.x_moteur, theta)
        self.R = np.tile(self.r_moteur, (Ntheta, 1))

        # Coordonnées en 3D
        self.Y = self.R * np.cos(Theta)
        self.Z = self.R * np.sin(Theta)
        if affichage:
            # Tracé 3D
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.X, self.Y, self.Z, alpha=1.0)
            ax.set_xlabel("Longueur (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title("Géométrie 3D du moteur-fusée")
            ax.view_init(elev=20, azim=120)
            plt.tight_layout()
            plt.show()

    def affichage_forme_generale(self, Ntheta=50, affichage=True):
        # Génére les profils
        self.forme_generale_2D(affichage=False)
        self.forme_generale_3D(Ntheta=Ntheta, affichage=False)

        # Création figure avec 2 sous-figures : une 2D et une 3D
        fig = plt.figure(figsize=(10, 8))

        # 1. Affichage 2D
        ax1 = fig.add_subplot(2, 1, 1)  # 2 lignes, 1 colonne, premier subplot
        ax1.plot(self.x_moteur, self.r_moteur, color="grey", label="Forme générale")
        ax1.plot(self.x_moteur, -self.r_moteur, color="grey")
        ax1.set_xlabel("Longueur (m)")
        ax1.set_ylabel("Rayon (m)")
        ax1.set_title("Géométrie moteur fusée (2D)")
        ax1.legend()
        ax1.axis("equal")

        # 2. Affichage 3D
        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        ax2.plot_surface(self.X, self.Y, self.Z, alpha=1.0)
        ax2.set_xlabel("Longueur (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")
        ax2.set_title("Géométrie moteur fusée (3D)")
        ax2.view_init(elev=20, azim=120)

        plt.tight_layout()

        if affichage:
            plt.show()

        return fig

class dimensionnement_moteur:
    def __init__(self, Q, P, T_chambre, gamma,r,ratio_chambre,alpha=0):
        self.ts = 15e-3  # temps de combustion
        # self.S_chambre = np.pi*(Q*self.ts/(np.pi*ratio_chambre))**(2/3)
        self.Q = Q # debit massique (kg/s)
        self.P = P*1e5 # pression dans la chambre (Pa)
        self.T_chambre = T_chambre
        self.gamma = gamma
        self.r = r
        self.R = 8.314
        self.alpha = alpha

        self.S_chambre = np.pi*(Q*self.ts/self.P*T_chambre*self.R/ratio_chambre/np.pi)**(2/3)

        self.M0 = self.mac_chambre() # mac dans la chambre
        # self.Rc = self.surface_rayon(self.S_chambre*self.rapport_col_Surface(self.M0)) #rayon col (m)
        # self.Rc = np.sqrt(self.S_chambre/(np.pi*ratio_col)) # rayon col (m)
        self.Rc = self.surface_rayon(Q*np.sqrt(T_chambre)/self.P*np.sqrt(r/gamma)*((gamma+1)/2)**((gamma+1)/(2*(gamma-1))))
        self.Me = self.mac(self.P,101325) # mac en sortie de tuyere
        # self.Re = self.Rc/np.sqrt(self.rapport_col_Surface(self.Me)) # rayon tuyere (m)
        self.Re = self.Rc/self.surface_rayon(self.rapport_col_Surface(self.Me)) # rayon tuyere (m)
 
        self.forme = geometrie(Q, Rc=self.Rc, Re=self.Re, S_chambre=self.S_chambre,ratio_chambre=ratio_chambre,ts=self.ts,alpha=alpha)
        self.affichage_mach(False)
        self.affichage_pression(False)
        self.affichage_temperature(False)
        self.Pousse = Q*np.sqrt(2*gamma/(gamma-1)*r*T_chambre*(1-(1/P)**((gamma-1)/gamma)))/9.81

    def mac_chambre(self):
        return self.Q/(self.P*self.S_chambre)*np.sqrt(self.r*self.T_chambre/self.gamma)

    def rapport_col_Surface(self,M):
        return M*((2+(self.gamma-1)*M**2)/(1+self.gamma))**(-(self.gamma+1)/(2*(self.gamma-1)))

    def surface_rayon(self,S):
        return np.sqrt(S/np.pi)
   
    def mac(self,Pch,Psection):
        return np.sqrt(2/(self.gamma-1)*((Pch/Psection)**((self.gamma-1)/self.gamma)-1))

    def rapport_pression(self,M):
        return (1+(self.gamma-1)/2*M*M)**(-self.gamma/(self.gamma-1))

    def rapport_temperature(self,M):
        return (1+(self.gamma-1)/2*M*M)**(-1)

    def mach_from_surface_ratio(self, S_ratio, supersonique=True):
        M_guess = 2.0 if supersonique else 0.002  # Choix de l’hypothèse initiale
        func = lambda M: 1/self.rapport_col_Surface(M) - S_ratio
        M_sol = fsolve(func, M_guess)[0]
        return M_sol
    
    def affichage_mach(self,affichage=True):
        Mmoteur = []
        if self.alpha ==0 :
            for R in self.forme.y_chambre:
                Mmoteur.append(self.mach_from_surface_ratio(R**2/self.Rc**2,supersonique=False))
            for R in self.forme.y_conv:
                Mmoteur.append(self.mach_from_surface_ratio(R**2/self.Rc**2,supersonique=False)) 
            for R in self.forme.y_tuyere:
                Mmoteur.append(self.mach_from_surface_ratio(R**2/self.Rc**2,supersonique=True))


        else :
            for R in self.forme.y_chambre:
                Mmoteur.append(self.mach_from_surface_ratio(R**2/self.Rc**2,supersonique=False))
            for R in self.forme.y_conv:
                Mmoteur.append(self.mach_from_surface_ratio(R**2/self.Rc**2,supersonique=False))
            for R in self.forme.y_col:
                Mmoteur.append(self.mach_from_surface_ratio(R**2/self.Rc**2,supersonique=True)) 
            for R in self.forme.y_tuyere:
                Mmoteur.append(self.mach_from_surface_ratio(R**2/self.Rc**2,supersonique=True))

        self.Mmoteur = np.array(Mmoteur)

        if affichage:
            plt.subplot(211)
            plt.plot(self.forme.x_moteur,Mmoteur)
            plt.subplot(212)
            plt.plot(self.forme.x_moteur,self.forme.r_moteur)
            plt.show()

    def affichage_temperature(self,affichage=True):
        Tmoteur = []
        for M in self.Mmoteur:
            Tmoteur.append(self.rapport_temperature(M)*self.T_chambre)
        self.Tmoteur = np.array(Tmoteur)
        if affichage:
            plt.subplot(211)
            plt.plot(self.forme.x_moteur,self.Tmoteur)
            plt.subplot(212)
            plt.plot(self.forme.x_moteur,self.forme.r_moteur)
            plt.show()

    def affichage_pression(self,affichage=True):
        Pmoteur = []
        for M in self.Mmoteur:
            Pmoteur.append(self.rapport_pression(M)*self.P)
        self.Pmoteur = np.array(Pmoteur)*1e-5
        if affichage:
            plt.subplot(211)
            plt.plot(self.forme.x_moteur,self.Pmoteur)
            plt.subplot(212)
            plt.plot(self.forme.x_moteur,self.forme.r_moteur)
            plt.show()
    
    def affichage_2D(self,affichage=True):

        fig,ax = plt.subplots(2, 2, figsize=(10, 6))
        ax[0,0].plot(self.forme.x_moteur,self.forme.r_moteur)
        ax[0,0].set_ylabel("Forme")

        ax[0,1].plot(self.forme.x_moteur,self.Mmoteur)
        ax[0,1].set_ylabel("Mach")

        ax[1,0].plot(self.forme.x_moteur,self.Pmoteur)
        ax[1,0].set_ylabel("Pression")

        ax[1,1].plot(self.forme.x_moteur,self.Tmoteur)
        ax[1,1].set_ylabel("Température")

        if affichage:
            plt.show()
        else :
            return fig

    def affichage_temperature_3D(self, N_theta=50):

        # 2. Révolution de la géométrie autour de x
        theta = np.linspace(0, 2*np.pi, N_theta)
        X, Theta = np.meshgrid(self.forme.x_moteur, theta)
        R = np.tile(self.forme.r_moteur, (len(theta), 1))
        
        # Coordonnées cylindriques vers cartésiennes
        Y = R * np.cos(Theta)
        Z = R * np.sin(Theta)

        # 3. Création du mapping de couleur
        T_norm = (self.Tmoteur - self.Tmoteur.min()) / (self.Tmoteur.max() - self.Tmoteur.min())  # Normalisation 0–1
        T_colors = plt.cm.plasma(T_norm)  # Colormap plasma
        facecolors = np.tile(T_colors, (len(theta), 1, 1))  # réplication dans l’axe theta

        # 4. Affichage
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("Température à la surface de la tuyère (3D)")
        mappable = plt.cm.ScalarMappable(cmap="plasma")
        mappable.set_array(self.Tmoteur)
        plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Température (K)")

        plt.tight_layout()
        plt.show()
    
    def affichage_pression_3D(self, N_theta=50):

        # 2. Révolution de la géométrie autour de x
        theta = np.linspace(0, 2*np.pi, N_theta)
        X, Theta = np.meshgrid(self.forme.x_moteur, theta)
        R = np.tile(self.forme.r_moteur, (len(theta), 1))
        
        # Coordonnées cylindriques vers cartésiennes
        Y = R * np.cos(Theta)
        Z = R * np.sin(Theta)

        # 3. Création du mapping de couleur
        T_norm = (self.Pmoteur - self.Pmoteur.min()) / (self.Pmoteur.max() - self.Pmoteur.min())  # Normalisation 0–1
        T_colors = plt.cm.plasma(T_norm)  # Colormap plasma
        facecolors = np.tile(T_colors, (len(theta), 1, 1))  # réplication dans l’axe theta

        # 4. Affichage
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("Pression à la surface de la tuyère (3D)")
        mappable = plt.cm.ScalarMappable(cmap="plasma")
        mappable.set_array(self.Pmoteur)
        plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Pression (bar)")

        plt.tight_layout()
        plt.show()
    
    def affichage_mach_3D(self, N_theta=50):

        # 2. Révolution de la géométrie autour de x
        theta = np.linspace(0, 2*np.pi, N_theta)
        X, Theta = np.meshgrid(self.forme.x_moteur, theta)
        R = np.tile(self.forme.r_moteur, (len(theta), 1))
        
        # Coordonnées cylindriques vers cartésiennes
        Y = R * np.cos(Theta)
        Z = R * np.sin(Theta)

        # 3. Création du mapping de couleur
        T_norm = (self.Mmoteur - self.Mmoteur.min()) / (self.Mmoteur.max() - self.Mmoteur.min())  # Normalisation 0–1
        T_colors = plt.cm.plasma(T_norm)  # Colormap plasma
        facecolors = np.tile(T_colors, (len(theta), 1, 1))  # réplication dans l’axe theta

        # 4. Affichage
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("Mach à la surface de la tuyère (3D)")
        mappable = plt.cm.ScalarMappable(cmap="plasma")
        mappable.set_array(self.Mmoteur)
        plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Pression (Pa)")

        plt.tight_layout()
        plt.show()

    def affichage_3D(self, N_theta=50):

        # Révolution de la géométrie
        theta = np.linspace(0, 2 * np.pi, N_theta)
        X, Theta = np.meshgrid(self.forme.x_moteur, theta)
        R = np.tile(self.forme.r_moteur, (len(theta), 1))
        Y = R * np.cos(Theta)
        Z = R * np.sin(Theta)

        # Normalisation des champs
        T_norm = (self.Tmoteur - self.Tmoteur.min()) / (self.Tmoteur.max() - self.Tmoteur.min())
        P_norm = (self.Pmoteur - self.Pmoteur.min()) / (self.Pmoteur.max() - self.Pmoteur.min())
        M_norm = (self.Mmoteur - self.Mmoteur.min()) / (self.Mmoteur.max() - self.Mmoteur.min())

        T_colors = np.tile(plt.cm.plasma(T_norm), (len(theta), 1, 1))
        P_colors = np.tile(plt.cm.viridis(P_norm), (len(theta), 1, 1))
        M_colors = np.tile(plt.cm.inferno(M_norm), (len(theta), 1, 1))

        # Affichage
        fig = plt.figure(figsize=(18, 12))

        # Température
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(X, Y, Z, facecolors=T_colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax1.set_title("Température (K)")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_zlabel("z (m)")
        mappable_T = plt.cm.ScalarMappable(cmap="plasma")
        mappable_T.set_array(self.Tmoteur)
        fig.colorbar(mappable_T, ax=ax1, shrink=0.5)

        # Pression
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(X, Y, Z, facecolors=P_colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax2.set_title("Pression (bar)")
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")
        ax2.set_zlabel("z (m)")
        mappable_P = plt.cm.ScalarMappable(cmap="viridis")
        mappable_P.set_array(self.Pmoteur)
        fig.colorbar(mappable_P, ax=ax2, shrink=0.5)

        # Mach
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_surface(X, Y, Z, facecolors=M_colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax3.set_title("Nombre de Mach")
        ax3.set_xlabel("x (m)")
        ax3.set_ylabel("y (m)")
        ax3.set_zlabel("z (m)")
        mappable_M = plt.cm.ScalarMappable(cmap="inferno")
        mappable_M.set_array(self.Mmoteur)
        fig.colorbar(mappable_M, ax=ax3, shrink=0.5)

        return fig

class moteur_reel:
    def __init__(self, moteur_avp, Q, P, T_chambre, gamma,r):
        self.moteur_avp = moteur_avp
        self.forme = self.moteur_avp.forme

        self.T_chambre = T_chambre
        self.P = P*1e5
        self.Q = Q

    def surface_rayon(self,S):
        return np.sqrt(S/np.pi)
    
    def rayon_col(self,Q,P,T):
        return self.surface_rayon(Q*np.sqrt(T)/P*np.sqrt(r/gamma)*((gamma+1)/2)**((gamma+1)/(2*(gamma-1))))
    
    def domaine_Rc(self,N=50,a=0.2):
        l_T = np.random.uniform(self.T_chambre*(1-a),self.T_chambre*(1+a),N)
        l_P = np.random.uniform(self.P*(1-a),self.P*(1+a),N)
        l_Q = np.random.uniform(self.Q*(1-a),self.Q*(1+a),N)

        points = []
        infos = []

        for i in range(N):

            x = l_Q[i]
            y = self.rayon_col(l_Q[i],l_P[i],l_T[i])
            points.append([x, y])
            infos.append((x, y, f"Q={l_Q[i]:.0f}, P={l_P[i]:.3f}, T={l_T[i]:.1f}"))

            plt.plot(x, y, 'o', color='gray', markerfacecolor='none')  # point vide

        # Calcul de l'enveloppe convexe
        points = np.array(points)
        hull = ConvexHull(points)

        # Tracer l’enveloppe
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r--', lw=2)

        # Affichage du texte pour les points de l’enveloppe
        for i in hull.vertices:
            x, y, label = infos[i]
            plt.text(x, y, label, fontsize=8, ha='left', va='bottom', rotation=0)

        plt.xlabel("debit (kg/s)")
        plt.ylabel("Rayon au col (m)")
        plt.title("Enveloppe convexe avec annotation des points extrêmes")
        plt.show()

class Thermique:
    def __init__(self, Diametre, vitesse=1, epaisseur=0.01, Tc=2500, T_ext=300, matiere="aluminium", carburant="essence"):
        # --- Propriétés matériaux solides (ex: aluminium) ---
        matiere = self.propriete_solide(matiere)
        self.cp_solide = matiere["cp"]      # J/kg/K
        self.rho_solide = matiere["rho"]     # kg/m³
        self.k_solide = matiere["k"]        # W/m/K

        # --- Géométrie et conditions ---
        self.D = Diametre          # diamètre du tube (m)
        self.u = vitesse           # vitesse gaz (m/s)
        self.e = epaisseur         # épaisseur paroi (m)
        self.T_gaz = Tc            # température gaz interne (K)
        self.T_ext = T_ext         # température externe (K)

        # --- Propriétés gaz brûlés selon carburant ---
        gaz = self.propriete_gaz(carburant)
        self.mu_gaz = gaz["mu"]
        self.rho_gaz = gaz["rho"]
        self.cp_gaz = gaz["cp"]
        self.k_gaz = gaz["k"]

        # --- Coefficients de transfert thermique ---
        self.h_int = self.colburn_h(self.rho_gaz,vitesse,self.D,self.mu_gaz,self.cp_gaz,self.k_gaz)   # h intérieur gaz/paroi
        # self.h_int = 10000
        self.h_ext = 1000               # h extérieur (ex: air ambiant forcé)
        self.h_ext = self.colburn_h(1000,5,self.D,1e-3,4184,0.598)

        # --- Propriétés auxiliaires ---
        self.alpha_solide = self.k_solide / (self.rho_solide * self.cp_solide)  # diffusivité thermique paroi

    # --- Calcul du coefficient Colburn pour l'intérieur ---
    def colburn_h(self,rho,u,D,mu,cp,k):
        Re = self.Reynolds(rho, u, D, mu)
        Pr = self.Prandtl(mu, cp, k)
        Nu = 0.023 * (Re ** 0.8) * (Pr ** 0.3)
        h = Nu * k / D
        return h

    def Reynolds(self, rho, u, D, mu):
        return rho * u * D / mu

    def Prandtl(self, mu, cp, k):
        return mu * cp / k

    def viscosite_sutherland(self, T, mu0=1.716e-5, T0=273.15, S=110.4):
        return mu0 * (T/T0)**1.5 * (T0 + S) / (T + S)

    def propriete_gaz(self, carburant):
        if carburant == "essence":
            return {"mu": 4.5e-5, "rho": 0.35, "cp": 1500, "k": 0.09}
        elif carburant == "CH4":
            return {"mu": 4.2e-5, "rho": 0.32, "cp": 1600, "k": 0.1}
        else:
            raise ValueError("Carburant non reconnu: essence ou CH4 attendus.")
    
    def propriete_solide(self, matiere):
        if matiere == "aluminium":
            return {"cp":897,"rho":2700,"k":185}
        elif matiere == "bois":
            # Cp de 1200 à 2700
            return {"cp":1500,"rho":850,"k":0.160}
        else:
            raise ValueError("matière non reconnu: aluminium attendus.")

    # --- Simulation de l'évolution temporelle de la température ---
    def evolution_tempo_temperature(self, N=100, return_data=False):
        dx = self.e / (N - 1)
        dt = 0.01  # pas de temps
        t_end = 5  # durée de l'animation en secondes

        # Conditions initiales
        T = np.ones(N) * self.T_ext
        T_evolution = []

        r = self.alpha_solide * dt / dx**2

        # Matrice A et vecteur B
        A = np.zeros((N, N))
        B = np.zeros(N)

        for i in range(1, N-1):
            A[i, i-1] = -r
            A[i, i]   = 1 + 2*r
            A[i, i+1] = -r

        # Bords avec Biot
        Bi_int = self.h_int * dx / self.k_solide
        A[0, 0] = 1 + 2*r + 2*r*Bi_int
        A[0, 1] = -2*r
        B[0] = 2*r*Bi_int*self.T_gaz

        Bi_ext = self.h_ext * dx / self.k_solide
        A[-1, -1] = 1 + 2*r + 2*r*Bi_ext
        A[-1, -2] = -2*r
        B[-1] = 2*r*Bi_ext*self.T_ext

        # Résolution
        temps = np.arange(0, t_end+dt, dt)
        for t in temps:
            T = np.linalg.solve(A, T + B)
            T_evolution.append(T.copy())

        # --- Animation ---
        X = np.linspace(0, self.e, N)

        if return_data:
            X = np.linspace(0, self.e, N)
            return T, X, T_evolution, temps
        else:

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios':[4,1]})
            
            # Graphique de température
            line, = ax1.plot(X*1000, T_evolution[0], lw=2)
            ax1.set_xlim(0, self.e*1000)
            ax1.set_ylim(self.T_ext - 50, self.T_gaz + 100)
            ax1.set_xlabel("Épaisseur x [mm]")
            ax1.set_ylabel("Température [K]")

            # Progression : fond gris clair
            ax2.set_xlim(0, t_end)
            ax2.set_ylim(0, 1)
            ax2.set_facecolor("#f0f0f0")  # fond gris clair
            ax2.get_yaxis().set_visible(False)  # cacher l'axe y
            ax2.set_xlabel("Temps [s]")  # label sous la barre
            ax2.set_xticks([0, t_end])
            ax2.set_xticklabels(["0 s", f"{t_end} s"])

            # Barre verte
            progress_bar, = ax2.plot([0,0], [0.5,0.5], color='green', lw=8)

            def update(i):
                line.set_ydata(T_evolution[i])
                ax1.set_title(f"Évolution de la température de la parois - t = {i*dt} s")
                
                # Mise à jour de la barre de progression
                progress_bar.set_data([0, i*dt], [0.5, 0.5])
                return line, progress_bar

            ani = animation.FuncAnimation(
                fig, update, frames=len(T_evolution), blit=True, interval=dt*1000
            )

            plt.tight_layout()
            plt.show()

reaction = chimie("essence",True,1)
pptchimie, cp, r, gamma, RMC = reaction.proprietes()

# flamme = Combustion(pptchimie)

# print(flamme.Tf)
# # # reaction.T_flamme_phi(phi_max=2,N=10)

Q = 0.05
P = 4
Tf = 2400

# forme1 = geometrie(Q, Rc=0.02, Re=0.05,alpha=0)
# forme2 = geometrie(Q, Rc=0.02, Re=0.05,alpha=0.2)
# forme1.forme_generale_2D()
# forme2.forme_generale_2D()
# plt.show()

# print(1)
# forme1 = dimensionnement_moteur(Q, P, Tf, gamma,r,7,alpha=0)
# print("affichage 1")
# forme1.affichage_2D(False)
# print(2)
# forme2 = dimensionnement_moteur(Q, P, Tf, gamma,r,7,alpha=0.2)
# print("affichage 2")
# forme2.affichage_2D(False)
# plt.show()

# moteur = dimensionnement_moteur(Q, P, Tf, gamma,r,7)
# # moteur.forme.forme_generale_2D()
# # # moteur.forme.col()
# # moteur.affichage_mach()
# moteur_r = moteur_reel(moteur,Q, P, Tf, gamma,r)
# # moteur_r.domaine_Rc(a = 0.1,N = 2000)

app = Application()
app.mainloop()



# ani = Thermique(2*moteur.forme.r_chambre,moteur.M0*np.sqrt(gamma*r*Tf)).evolution_tempo_temperature(100)

