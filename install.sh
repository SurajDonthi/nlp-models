conda env create -f environment.yml
eval "$(conda shell.bash hook)"
conda activate NLPDev
echo "Activated NLPDev"
echo "Installing spaCy models..."
spacy download en_core_web_md
spacy download en_core_web_sm
spacy download en
echo "Installing neuralcoref..."
git clone https://github.com/huggingface/neuralcoref.git ~/neuralcoref
cd ~/neuralcoref/
pip install -r requirements.txt
pip install -e .
cd -
# echo "Deleting neuralcoref folder..."
# rm -r ./neuralcoref

wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O drivers/google-chrome-stable_current_amd64.deb
sudo dpkg -i drivers/google-chrome-stable_current_amd64.deb
wget https://chromedriver.storage.googleapis.com/89.0.4389.23/chromedriver_linux64.zip -O drivers/chromedriver_linux64.zip
unzip drivers/chromedriver_linux64.zip -d drivers/
mv drivers/chromedriver /usr/bin/chromedriver
chown root:root /usr/bin/chromedriver
chmod +x /usr/bin/chromedriver