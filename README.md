# sompz
`SOMPZ` is a redshift calibration method using Self-Organizing Maps (SOMs) to compute the redshift distributions of galaxies in tomographic bins leveraging information from smaller but deeper fields. 

### Setup Anaconda environment

`module load python`

`module load cray-fftw gcc`

This seems to be required to setup conda (see https://github.com/des-science/y3-wl_image_sims/blob/master/setup_nersc.sh)

`. /usr/common/software/python/2.7-anaconda-4.4/etc/profile.d/conda.sh`

`conda create --name py27-sompz --file conda_env_explicit.txt`

`conda activate py27-sompz`

`pip install -r requirements.txt`

You can then use the `sompz` conda environment by running `source activate py27-sompz`. You will need to activate this environment each time you want to use it, or automate this process by including the command in a file like `.bashrc`

If you plan to use this environment on jupyter.nersc.gov, you will need to run this command once from `cori`:

`ipython kernel install --name py27-sompz --user`

You should now see a `py27-sompz` option in your list of kernels on JupyterHub.