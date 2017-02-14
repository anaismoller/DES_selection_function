from astropy.cosmology import FlatLambdaCDM
import argparse

cosmo = FlatLambdaCDM(H0=70, Om0=0.27)

def dist_mu(redshift):
    mu = cosmo.distmod(redshift)

    return mu.value

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-z", "--z")
    args = parser.parse_args()
    redshift = float(args.z)

    dist_mu(redshift)
