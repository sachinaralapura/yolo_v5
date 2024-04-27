**Install Flask**

```bash
pip install flask
```

**Create virtual environment and activate**

```bash
python -m venv .env 

source .env/bin/activate.bash
```

**Clone repo and install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.8.**

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install

```
**Run flask application**

```bash
flask --app app run
```