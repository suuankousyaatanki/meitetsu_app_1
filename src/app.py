# 必要なモジュールのインポート
import torch
from meitetsu import transform, Net # meitetsu.pyから前処理とネットワークの定義を読み込み
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

# 学習済みモデルをもとに推論する
def predict(img):
    # ネットワークの定義
    net = Net().cpu().eval()
    # 学習済みモデルの重み
    net.load_state_dict(torch.load("./meitetsu_ver1 (1).pt", map_location=torch.device("cpu")))
    # データの前処理
    img = transform(img)
    img = img.unsqueeze(0) # 1次元増やす
    # 推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y


# 推論したラベルから特急かそうでないかを返す関数
def getName(label):
    if label==0:
        return "特急"
    elif label==1:
        return "特急じゃない"


# Flaskのインスタンスを作成
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(["png", "jpg", "gif", "jpeg"])

# 拡張子が適切かどうかをチェック
def allwed_file(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


# URLにアクセスがあった場合の挙動の設定
@app.route("/", methods = ["GET", "POST"])
def predicts():
    # リクエストがポストがどうかの判別
    if request.method == "POST":
        # ファイルがなかったときの処理
        if "filename" not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files["filename"]
        # ファイルのチェック
        if file and allwed_file(file.filename):
            # 画像ファイルに対する処理
            # 画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert("RGB")
            # 画像データをバッファを書き込む
            image.save(buf, "png")
            # バイナリデータをbase64でエンコードしてutf-8でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            # HTML側のsrcの記述に合わせるために付帯情報を付与する
            base64_data = "data:image/png;base64,{}".format(base64_str)
            # 入力された画像に対して推論
            pred = predict(image)
            meitetsuName_ = getName(pred)
            return render_template("result.html", meitetsuName=meitetsuName_, image=base64_data)
        return redirect(request.url)


    # GETメソッドの定義
    elif request.method == "GET":
        return render_template("index.html")


# アプリケーションの実行の定義
if __name__ == "__main__":
    app.run(debug=True)
