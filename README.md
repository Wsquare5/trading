# Deepseek API 调用

是否受够了 deepseek-r1 的服务器繁忙？不如让我们直接调用 deepseek API 来聊天吧。

## 项目设置

### 1. 环境准备

首先创建并激活虚拟环境：

```bash
#创建虚拟环境
python -m venv venv
#激活虚拟环境
#Windows:
.\venv\Scripts\activate
#macOS/Linux:
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 环境变量配置

在项目根目录创建 `.env` 文件，添加以下内容：

```
DASHSCOPE_API_KEY=your_api_key_here
```

## 获取阿里云百炼 API 密钥

1. 访问[阿里云百炼平台](https://dashscope.aliyun.com/)
2. 注册/登录阿里云账号
3. 在控制台中找到"API 密钥管理"
   - 点击右上角的"个人信息"
   - 选择"API 密钥管理"
   - 如果没有密钥，点击"创建 API 密钥"
4. 复制 API 密钥并保存到 `.env` 文件中

## 运行应用

```bash
python app.py
```

访问 http://localhost:8080 即可使用应用。

## 项目结构

```
.
├── app.py              # Flask 应用主文件
├── requirements.txt    # 项目依赖
├── .env               # 环境变量配置
├── static/            # 静态文件
│   └── style.css      # 样式文件
└── templates/         # 模板文件
    └── index.html     # 主页模板
```

## 注意事项

- 请确保 `.env` 文件已经正确配置 API 密钥
- 不要将 API 密钥提交到版本控制系统
- 确保虚拟环境已激活再运行应用

## 常见问题解决

1. 如果遇到模块未找到的错误，请确认：
   - 虚拟环境是否已激活
   - 所有依赖是否已安装
2. 如果遇到 API 调用失败，请检查：
   - API 密钥是否正确配置
   - 网络连接是否正常
   - API 调用额度是否充足
