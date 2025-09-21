import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 環境変数の読み込み
load_dotenv()

def get_expert_system_message(expert_type):
    """専門家タイプに応じてシステムメッセージを返す"""
    system_messages = {
        "プログラミング専門家": """あなたは経験豊富なソフトウェア開発者です。
        プログラミング、ソフトウェア設計、デバッグ、最新の技術トレンドに関する質問に、
        実践的で具体的なアドバイスを提供してください。
        コード例や具体的な解決策を含めた回答を心がけてください。""",
        
        "医療専門家": """あなたは医療分野の専門知識を持つアドバイザーです。
        健康、医療、予防に関する一般的な情報を提供してください。
        ただし、具体的な診断や治療については医師に相談するよう案内し、
        一般的な健康情報と教育的な内容に焦点を当ててください。""",
        
        "教育専門家": """あなたは教育分野の専門家です。
        学習方法、教育理論、スキル向上、知識習得に関するアドバイスを提供してください。
        効果的な学習戦略や教育リソースについて、
        根拠に基づいた実践的なアドバイスを心がけてください。"""
    }
    return system_messages.get(expert_type, "あなたは親切で知識豊富なアシスタントです。")

def get_llm_response(user_input, expert_type):
    """
    入力テキストと専門家選択値を受け取り、LLMからの回答を返す
    
    Args:
        user_input (str): ユーザーの入力テキスト
        expert_type (str): 選択された専門家のタイプ
    
    Returns:
        str: LLMからの回答
    """
    try:
        # OpenAI APIキーの確認
        if not os.getenv("OPENAI_API_KEY"):
            return "エラー: OPENAI_API_KEYが設定されていません。.envファイルに設定してください。"
        
        # ChatOpenAIの初期化
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # システムメッセージとユーザーメッセージの作成
        system_message = SystemMessage(content=get_expert_system_message(expert_type))
        human_message = HumanMessage(content=user_input)
        
        # LLMに送信
        messages = [system_message, human_message]
        response = llm.invoke(messages)
        
        return response.content
        
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

# Streamlitアプリのメイン部分
def main():
    st.title("🤖 AI専門家チャット")
    st.write("専門家を選択して質問してください。AIが選択した分野の専門家として回答します。")
    
    # 専門家選択のラジオボタン
    expert_options = ["プログラミング専門家", "医療専門家", "教育専門家"]
    selected_expert = st.radio(
        "専門家を選択してください:",
        expert_options,
        index=0
    )
    
    # 選択された専門家の説明を表示
    expert_descriptions = {
        "プログラミング専門家": "💻 ソフトウェア開発、プログラミング、技術に関する質問にお答えします。",
        "医療専門家": "🏥 健康、医療、予防に関する一般的な情報を提供します。",
        "教育専門家": "📚 学習方法、教育理論、スキル向上に関するアドバイスを提供します。"
    }
    
    st.info(expert_descriptions[selected_expert])
    
    # 入力フォーム
    user_input = st.text_area(
        "質問を入力してください:",
        height=100,
        placeholder="ここに質問を入力してください..."
    )
    
    # 送信ボタン
    if st.button("質問を送信", type="primary"):
        if user_input.strip():
            # ローディング表示
            with st.spinner(f"{selected_expert}が回答を準備中..."):
                response = get_llm_response(user_input, selected_expert)
            
            # 回答の表示
            st.subheader("💡 回答:")
            st.write(response)
            
            # セッションステートに履歴を保存
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append({
                "expert": selected_expert,
                "question": user_input,
                "answer": response
            })
            
        else:
            st.warning("質問を入力してください。")
    
    # チャット履歴の表示
    if "chat_history" in st.session_state and st.session_state.chat_history:
        with st.expander("📜 チャット履歴", expanded=False):
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                st.write(f"**{chat['expert']}への質問 #{len(st.session_state.chat_history)-i}:**")
                st.write(f"Q: {chat['question']}")
                st.write(f"A: {chat['answer']}")
                st.divider()

if __name__ == "__main__":
    main()