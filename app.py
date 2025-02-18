from flask import Flask, render_template, request, jsonify
import os
import dashscope

app = Flask(__name__)

# 存储所有对话的字典，key 是会话 ID
conversations = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id', 'default')
    message = data.get('message')
    
    # 如果是新会话，初始化消息列表
    if session_id not in conversations:
        conversations[session_id] = []
    
    # 添加用户消息
    conversations[session_id].append({
        'role': 'user',
        'content': message
    })
    
    try:
        # 调用 DeepSeek API
        response = dashscope.Generation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model="deepseek-r1",
            messages=conversations[session_id],
            result_format='message'
        )
        
        # 获取回复
        assistant_message = response.output.choices[0].message.content
        reasoning = response.output.choices[0].message.reasoning_content
        
        # 添加助手回复到对话历史
        conversations[session_id].append({
            'role': 'assistant',
            'content': assistant_message
        })
        
        return jsonify({
            'status': 'success',
            'response': assistant_message,
            'reasoning': reasoning
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 