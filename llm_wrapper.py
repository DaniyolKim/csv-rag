import os
import requests
import json
from typing import List, Dict, Optional

class LLMClient:
    def __init__(self, provider="ollama", model="exaone3.5:32b"):
        self.provider = provider
        self.model = model
        
        if provider == "ollama":
            # Ollama는 기본적으로 localhost:11434에서 실행됩니다
            self.base_url = "http://localhost:11434/api"
        else:
            raise ValueError(f"지원되지 않는 LLM 제공자입니다: {provider}")
    
    def get_available_models(self) -> List[str]:
        """Ollama에서 사용 가능한 모델 목록을 반환합니다."""
        if self.provider != "ollama":
            raise ValueError("이 메서드는 Ollama 제공자에서만 사용할 수 있습니다")
        
        try:
            response = requests.get(f"{self.base_url}/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                return []
        except Exception as e:
            print(f"모델 목록을 가져오는 중 오류 발생: {str(e)}")
            return []

    def chat(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """LLM에 메시지를 보내고 응답을 받습니다."""
        if self.provider == "ollama":
            # Ollama API 형식으로 변환
            ollama_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                ollama_messages.append({"role": role, "content": content})
            
            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {"temperature": temperature}
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/chat",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload)
                )
                
                if response.status_code == 200:
                    return response.json().get("message", {}).get("content", "")
                else:
                    return f"오류: {response.status_code} - {response.text}"
            except Exception as e:
                return f"API 호출 중 오류 발생: {str(e)}"
        
        else:
            raise ValueError("지원되지 않는 LLM 제공자입니다")
