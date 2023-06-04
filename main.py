
import plots as pl
import softmax as sm
import vector as vec
import softmax as sm
import pca as pca
import kmeans as km
import dataf as dt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nayve as nv
import dataNaive as dn

def main():

    teams = dt.dataf()

    option = None
    while True:
        print("1 - Bar charts")
        print("2 - Stack Bar Charts")
        print("3 - Exit")
        option = int(input("Choose an option: "))
        if option < 1 or option > 3:
            print("Invalid option")
            continue
        if option == 3:
            break
        if option == 1:
            pl.barCharts(teams)
            break
        if option == 2:
            pl.stackedBars(teams)
            break


    vec.vector()

    print("Softmax")

    teams = sm.softmax()

    while True:
        print("1 - Bar charts")
        print("2 - Stack Bar Charts")
        print("3 - Exit")
        option = int(input("Choose an option: "))
        if option < 1 or option > 3:
            print("Invalid option")
            continue
        if option == 3:
            break
        if option == 1:
            pl.barCharts(teams, title="Softmax")
            break
        if option == 2:
            pl.stackedBars(teams, title="Softmax")
            break
    
    print("PCA")

    teams = pca.pca()

    while True:
        print("1 - Bar charts")
        print("2 - Stack Bar Charts")
        print("3 - Exit")
        option = int(input("Choose an option: "))
        if option < 1 or option > 3:
            print("Invalid option")
            continue
        if option == 3:
            break
        if option == 1:
            pl.barCharts(teams, title="PCA")
            break
        if option == 2:
            pl.stackedBars(teams, title="PCA")
            break

    km.kmeans()
    

    dn.dataNaive()

    print("Nayve")

    while True:
        print("1 - PCA")
        print("2 - No PCA")
        print("3 - Exit")
        option = int(input("Choose an option: "))
        if option < 1 or option > 3:
            print("Invalid option")
            continue
        if option == 3:
            break
        if option == 1:
            nv.nayve(isPCA=True)
            break
        if option == 2:
            nv.nayve(isPCA=False)
            break
    input("Press enter to exit...")
    return


if __name__ == "__main__":
    main()