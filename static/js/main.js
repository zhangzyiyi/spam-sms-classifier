document.addEventListener('DOMContentLoaded', function() {
    const smsForm = document.getElementById('smsForm');
    const resultDiv = document.getElementById('result');
    const resultAlert = document.getElementById('resultAlert');
    const predictionText = document.getElementById('predictionText');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');

    smsForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // 获取输入文本
        const text = document.getElementById('smsText').value;
        
        // 创建FormData对象
        const formData = new FormData();
        formData.append('text', text);
        
        // 发送预测请求
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // 显示结果
            resultDiv.style.display = 'block';
            
            // 设置预测结果样式
            resultAlert.className = 'alert ' + (data.label === 1 ? 'alert-spam' : 'alert-normal');
            
            // 更新预测文本
            predictionText.textContent = `预测结果：${data.prediction}`;
            
            // 更新置信度进度条
            const confidence = Math.round(data.confidence * 100);
            confidenceBar.style.width = `${confidence}%`;
            confidenceBar.className = `progress-bar ${data.label === 1 ? 'bg-danger' : 'bg-success'}`;
            confidenceBar.setAttribute('aria-valuenow', confidence);
            
            // 更新置信度文本
            confidenceText.textContent = `置信度：${confidence}%`;
            
            // 如果是垃圾短信且置信度高，添加警告图标
            if (data.label === 1 && confidence > 80) {
                predictionText.innerHTML = '⚠️ ' + predictionText.textContent;
            }
        })
        .catch(error => {
            resultDiv.style.display = 'block';
            resultAlert.className = 'alert alert-danger';
            predictionText.textContent = `错误：${error.message}`;
            confidenceBar.style.display = 'none';
            confidenceText.style.display = 'none';
        });
    });
}); 