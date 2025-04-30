from zhpr.predict import DocumentDataset, merge_stride, decode_pred
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import time
import os

class PunctuationRestoreModel:
    """
    中文標點符號恢復模型封裝類，保持模型常駐在GPU中
    """
    def __init__(self, model_name='p208p2002/zh-wiki-punctuation-restore', local_dir='./models/zh-punc'):
        self.model_name = model_name
        self.local_dir = local_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {self.device}")
        
        # 載入模型和tokenizer
        self.model, self.tokenizer = self._load_or_download_model()
        
        # 將模型移至GPU並設置為評估模式
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 預熱模型以初始化CUDA核心和緩存
        self._warmup()
        
    def _load_or_download_model(self):
        """嘗試從本地資料夾讀取模型；若不存在則自動從HuggingFace下載並儲存"""
        if os.path.exists(self.local_dir):
            print(f"✅ 從本地載入模型: {self.local_dir}")
            model = AutoModelForTokenClassification.from_pretrained(self.local_dir, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(self.local_dir, local_files_only=True)
        else:
            print(f"⬇️ 從HuggingFace下載模型: {self.model_name}")
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            print(f"💾 儲存模型到本地: {self.local_dir}")
            os.makedirs(self.local_dir, exist_ok=True)
            model.save_pretrained(self.local_dir)
            tokenizer.save_pretrained(self.local_dir)
            
        return model, tokenizer
    
    def _warmup(self):
        """預熱模型，初始化CUDA核心和緩存"""
        dummy_input = torch.randint(0, 1000, (1, 32)).to(self.device)
        with torch.no_grad():
            self.model(input_ids=dummy_input, attention_mask=torch.ones_like(dummy_input))
        print("🔥 模型預熱完成")
    
    def predict_step(self, batch_input_ids):
        """執行一個批次的預測"""
        batch_input_ids = batch_input_ids.to(self.device)
        
        # 創建注意力遮罩
        attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).long()
        
        # 使用no_grad來加速推理
        with torch.no_grad():
            output = self.model(input_ids=batch_input_ids, attention_mask=attention_mask)
        
        predicted_token_class_id_batch = output['logits'].argmax(-1)
        
        batch_out = []
        for predicted_token_class_ids, input_ids in zip(predicted_token_class_id_batch, batch_input_ids):
            # 找出padding tokens的起始位置
            input_ids_cpu = input_ids.cpu().tolist()
            try:
                input_id_pad_start = input_ids_cpu.index(self.tokenizer.pad_token_id)
            except ValueError:
                input_id_pad_start = len(input_ids_cpu)
            
            # 截斷到pad起始位置
            input_ids_cpu = input_ids_cpu[:input_id_pad_start]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids_cpu)
            
            # 獲取預測的標籤
            predicted_tokens_classes = [
                self.model.config.id2label[t.item()] 
                for t in predicted_token_class_ids[:input_id_pad_start].cpu()
            ]
            
            # 組合token和預測標籤
            out = list(zip(tokens, predicted_tokens_classes))
            batch_out.append(out)
            
        return batch_out
    
    def restore_punctuation(self, text, window_size=256, step=200, batch_size=8):
        """恢復文本中的標點符號"""
        start_time = time.time()
        
        # 將英文字詞放入(目前模型並不支持此斷句，會判定為[UNK]) 
        encoding = self.tokenizer(text,return_offsets_mapping=True)
        input_ids = encoding.input_ids
        unk_indexes = [i for i,id in enumerate(input_ids) if id == self.tokenizer.unk_token_id]
        unk_token_mapping = [encoding.offset_mapping[unk_i] for unk_i in unk_indexes] # offset_mapping 可以找回 tokenize 之前的文字
        unk_tokens = [text[unk_token_pos_range[0]:unk_token_pos_range[1]] for unk_token_pos_range in unk_token_mapping]
        # print(unk_tokens)
        
        # 創建數據集和數據加載器
        dataset = DocumentDataset(text, window_size=window_size, step=step)
        dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
        
        # 執行預測
        model_pred_out = []
        for batch in dataloader:
            batch_out = self.predict_step(batch)
            model_pred_out.extend(batch_out)
        
        # 合併結果
        merge_pred_result = merge_stride(model_pred_out, step)

        # 回補[UNK]原始字串
        for number, index in enumerate(unk_token_mapping):
            index_start = index[0]
            index_end = index [1]
            for word in range(index_start, index_end):
                tmp = list(merge_pred_result[word])
                tmp[0] = unk_tokens[number][word - index_start]
                merge_pred_result[word] = tuple(tmp)
        
        # 將剩下的[UNK]全部補為space
        for index in range(len(merge_pred_result)):
            if merge_pred_result[index][0] == '[UNK]':
                tmp = list(merge_pred_result[index])
                tmp[0] = ' '
                merge_pred_result[index] = tuple(tmp)
            else:
                pass

        merge_pred_result_decode = decode_pred(merge_pred_result)
        result_text = ''.join(merge_pred_result_decode)
        
        end_time = time.time()
        proc_time = end_time - start_time
        
        # print(f"處理時間: {proc_time:.2f} 秒")
        # print(f"文本長度: {len(text)} 字符")
        # print(f"處理速度: {len(text)/proc_time:.2f} 字符/秒")
        
        return result_text, proc_time

'''  
# 主程式入口點
if __name__ == "__main__":
    # 初始化模型（只需執行一次，之後模型會常駐在GPU中）
    model = PunctuationRestoreModel()
    
    # 示例文本
    test_text = (
        "本方案由瑞擎數位提供並搭配亞旭電腦的 5G 基站進行展示提升 5G 專"
        "網中UE的流量可視性與管理性通過 IP Mobility 與 GrismMEC 流量管理技術"
        "即時監控與控制流量確保企業網路安全 5G 專網與 SD-WAN 和PQC 加密技術相結合"
        "保障跨區域資料傳輸的機密性，並將流量 Log 與NetFlow 資訊(網路流量監控資訊)"
        "傳送至 SOC延續傳統企業網路的威脅偵測功能實現 5G 專網與企業網路的安全融合。"
    )

    # 執行標點符號恢復
    result, _ = model.restore_punctuation(test_text)
    print("結果:", result)

    # 演示模型已經在GPU中，第二次調用會更快
    print("\n再次處理相同文本...")
    result, _ = model.restore_punctuation(test_text)
    print("結果:", result)
    
    # 處理更長的文本
    long_text = test_text * 5
    print("\n處理更長文本...")
    result, _ = model.restore_punctuation(long_text)
    print("結果:", result[:100] + "...")
'''