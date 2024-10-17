# streamlit
- 教科書[輕量又漂亮的 Python Web 框架 - Streamlit AI 時代非學不可](https://www.tenlong.com.tw/products/9786267383988?list_name=srh)
- [CONTENT](CONTENT.MD)
# streamlit_1.py
```python
import streamlit as st
st.write('這是A888168的第一個Streamlit應用, Hello Mydeargreatteacher。')
```

# streamlit_4_1.py
```python
# 第四章/emoji.py
import streamlit as st

st.title("標題元素支援表情符號 :tada:")
st.header("章節元素支援表情表情 :apple:")
st.subheader("子章節元素支援表情符號，使用ASCII值插入\U0001F600")
st.code("代碼塊支援表情符號，使用ASCII值插入\U0001F602")
st.text("普通文本支援表情符號，使用ASCII值插入\U0001F601")
st.markdown("Markdown支援表情符號 :smile:")

```

# 
```python
# 第三章/line_chart.py
import streamlit as st
import pandas as pd

# 定義資料,以便創建資料框
data = {
    '月份':['01月', '02月', '03月'],
    '1號門店':[200, 150, 180],
    '2號門店':[120, 160, 123],
    '3號門店':[110, 100, 160],
}
# 根據上面創建的data，創建資料框
df = pd.DataFrame(data)
# 定義資料框所用的新索引
index = pd.Series([1, 2, 3,], name='序號')
# 將新索引應用到資料框上
df.index = index

st.header("A888168門店數據")
# 使用write()方法展示資料框
st.write(df)
st.header("折線圖")

st.subheader("設置x參數")
# 通過x指定月份所在這一列為折線圖的x軸
st.line_chart(df, x='月份')



# 修改df，用月份列作為df的索引，替換原有的索引
df.set_index('月份', inplace=True)

st.subheader("設置y參數")
# 通過y參數篩選只顯示1號門店的資料
st.line_chart(df, y='1號門店')
# 通過y參數篩選只顯示2、3號門店的資料
st.line_chart(df, y=['2號門店','3號門店'])

st.subheader("設置width、height和use_container_width參數")
# 通過width、height和use_container_width指定折線圖的寬度和高度
st.line_chart(df, width=300, height=300, use_container_width=False)
```

# 觀摩學習[Remove background from your image](https://bgremoval.streamlit.app/)
- [GITHUB](https://github.com/tyler-simons/BackgroundRemoval/tree/main)
```python
import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Remove background from your image")
st.write(
    ":dog: Try uploading an image to watch the background magically removed. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin:"
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = remove(image)
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    fix_image("./zebra.jpg")
```

# streamlit_1.py
```python

```

# streamlit_1.py
```python

```

# streamlit_1.py
```python

```

# [streamlit/llm-examples](https://llm-examples.streamlit.app/)
- [GITHUB](https://github.com/streamlit/llm-examples/tree/main)

# [Streamlit for Data Science](https://www.packtpub.com/en-tw/product/streamlit-for-data-science-9781803248226?srsltid=AfmBOoooJpKCpnO61p5z7ET-nNxY9PiQiPaSAM2f-sHUBDLc74n9zXZY)
- [github](https://github.com/tylerjrichards/Streamlit-for-Data-Science/blob/main/huggingface_demo/streamlit_app.py)
- [CONTENT1](CONTENT1.MD)
### Streamlit-for-Data-Science/huggingface_demo/streamlit_app.py

```
pip install huggingface_hub
pip install openai
```
```python
import openai
import streamlit as st
from transformers import pipeline

st.title("Hugging Face Demo")
text = st.text_input("Enter text to analyze")


@st.cache_resource()
def get_model():
    return pipeline("sentiment-analysis")


model = get_model()
if text:
    result = model(text)
    st.write("Sentiment:", result[0]["label"])
    st.write("Confidence:", result[0]["score"])

st.title("OpenAI Version")


openai.api_key = st.secrets["OPENAI_API_KEY"]

system_message_default = """You are a helpful sentiment analysis assistant. You always respond with the sentiment of the text you are given and the confidence of your sentiment analysis with a number between 0 and 1"""

system_message = st.text_area(
    "Enter a System Message to instruct OpenAI", system_message_default
)
analyze_button = st.button("Analyze Text")

if analyze_button:
    messages = [
        {
            "role": "system",
            "content": f"{system_message}",
        },
        {
            "role": "user",
            "content": f"Sentiment analysis of the following text: {text}",
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    sentiment = response.choices[0].message["content"].strip()
    st.write(sentiment)
```
