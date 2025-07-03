
pip install --upgrade pip  # ensures that pip is current

apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl
pip install -r requirements.txt
apt-get install curl git
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
pip install mecab

git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .

wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip