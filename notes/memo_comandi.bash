# --- CONTAINER --- 

# docker
docker build -t creazione_dataset .
docker run -it --name download_impactMesh-mbarborini-rsde -v /raid/home/rsde/mbarborini:/workspace creazione_dataset
docker start download_impactMesh-mbarborini-rsde
docker stop download_impactMesh-mbarborini-rsde


# rendere visibile path per earthengine-api
export PATH="$PATH:$HOME/.local/bin"

# autenticazione earthengine-api
earthengine authenticate

# git
git init # in /mbarborini (fuori dal container) solo la prima volta
git add .
git commit -m "messaggio commit"
git push

pip install --user -r requirements.txt

# --- LOCALE ---
python3 -m venv venv
source venv/bin/activate

pip freeze > requirements.txt

# jumphost da wsl e vscode 

./opkssh login
cp ~/.ssh/id_ecdsa* /mnt/c/Users/matti/.ssh/

# 🟢 🔴 🟡 🔵 ✔️ ❌
