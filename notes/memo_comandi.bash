# --- CONTAINER --- 

# docker
docker build -t creazione_dataset .
docker run -it --name download_impactMesh-mbarborini-rsde -v /raid/home/rsde/mbarborini:/workspace creazione_dataset


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

# 🟢 🔴 🟡 🔵 ✔️ ❌
