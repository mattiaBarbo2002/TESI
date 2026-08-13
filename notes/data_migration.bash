# ---> TERMINALE LOCALE

./dhcli.exe register -e rsde https://core.rsde.atlas.fbk.eu     # prima volta
./dhcli.exe login -e rsde

# se esistono già
rm config.env
rm credentials.env
rm env_variables.env

./dhcli.exe credentials > credentials.env
./dhcli.exe config > config.env
cat credentials.env config.env > env_variables.env

# copia
cat env_variables.env 
cat .dhcore.ini         # cat /mnt/c/Users/matti/.dhcore.ini



# ---> CLUSTER

rm .dhcore.ini
rm env_variables.env

# incolla
nano .dhcore.ini
nano env_variables.env

export DH_CONFIG=/raid/home/rsde/mbarborini.dhcore.ini
source ./set_env.sh env_variables.env
./dhcli list projects 

./dhcli upload artifact -p datasets -f (path cartella da copiare, pwd) -n nome_artifact







