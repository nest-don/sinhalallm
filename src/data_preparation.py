import pandas as pd
import json
import random
from typing import List, Dict
import re

class SinhalaDataProcessor:
    """Data processor for Sinhala LLM training data preparation"""
    
    def __init__(self):
        self.processed_data = []
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize Sinhala text"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).strip()
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def process_qa_data(self, filepath: str) -> List[Dict]:
        """Process question-answer CSV files"""
        print(f"Processing QA data from {filepath}")
        df = pd.read_csv(filepath)
        
        processed = []
        for _, row in df.iterrows():
            if 'Question' in df.columns and 'Answer' in df.columns:
                # Sinhala-QA format
                question = self.clean_text(row['Question'])
                answer = self.clean_text(row['Answer'])
                
                if question and answer:
                    processed.append({
                        'input': f"ප්‍රශ්නය: {question}",
                        'output': answer,
                        'instruction': "පහත ප්‍රශ්නයට සිංහලෙන් පිළිතුරු දෙන්න:",
                        'type': 'qa'
                    })
            
            elif 'sinhala_question' in df.columns and 'sinhala_answer' in df.columns:
                # Alternative QA format
                question = self.clean_text(row['sinhala_question'])
                answer = self.clean_text(row['sinhala_answer'])
                
                if question and answer:
                    processed.append({
                        'input': f"ප්‍රශ්නය: {question}",
                        'output': answer,
                        'instruction': "පහත ප්‍රශ්නයට සිංහලෙන් පිළිතුරු දෙන්න:",
                        'type': 'qa'
                    })
        
        print(f"Processed {len(processed)} QA pairs from {filepath}")
        return processed
    
    def process_summary_data(self, filepath: str) -> List[Dict]:
        """Process text-summary CSV files"""
        print(f"Processing summary data from {filepath}")
        df = pd.read_csv(filepath)
        
        processed = []
        for _, row in df.iterrows():
            if 'text' in df.columns and 'summary' in df.columns:
                text = self.clean_text(row['text'])
                summary = self.clean_text(row['summary'])
                
                if text and summary and len(text) > len(summary):
                    processed.append({
                        'input': f"පෙළ: {text}",
                        'output': summary,
                        'instruction': "පහත පෙළේ සාරාංශයක් සිංහලෙන් ලියන්න:",
                        'type': 'summarization'
                    })
        
        print(f"Processed {len(processed)} summary pairs from {filepath}")
        return processed
    
    def process_dolly_data(self, filepath: str) -> List[Dict]:
        """Process Dolly-style instruction data"""
        print(f"Processing Dolly data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        processed = []
        for _, row in df.iterrows():
            instruction = self.clean_text(row['instruction'])
            response = self.clean_text(row['response'])
            context = self.clean_text(row.get('context', ''))
            
            if instruction and response:
                input_text = instruction
                if context:
                    input_text = f"සන්දර්භය: {context}\n\nසූචනය: {instruction}"
                
                processed.append({
                    'input': input_text,
                    'output': response,
                    'instruction': instruction,
                    'type': 'instruction'
                })
        
        print(f"Processed {len(processed)} instruction pairs from {filepath}")
        return processed
    
    def create_conversation_format(self, data_point: Dict) -> str:
        """Convert data point to conversation format for training"""
        instruction = data_point['instruction']
        input_text = data_point['input']
        output = data_point['output']
        
        # Create a conversation format
        conversation = f"<|user|>\n{instruction}\n{input_text}\n<|assistant|>\n{output}<|end|>"
        return conversation
    
    def create_alpaca_format(self, data_point: Dict) -> Dict:
        """Convert data point to Alpaca format"""
        return {
            "instruction": data_point['instruction'],
            "input": data_point['input'],
            "output": data_point['output']
        }
    
    def prepare_training_data(self):
        """Prepare all training data"""
        print("Starting Sinhala LLM data preparation...")
        
        # Process all datasets
        qa_1k = self.process_qa_data('Sinhala-QA-1K.csv')
        qa_10k = self.process_qa_data('Sinhala-QA-10K.csv')
        summaries = self.process_summary_data('Sinhala text-Summary.csv')
        dolly = self.process_dolly_data('databricks-dolly-15k-sinhala.csv')
        
        # Combine all data
        all_data = qa_1k + qa_10k + summaries + dolly
        print(f"Total combined data points: {len(all_data)}")
        
        # Shuffle data
        random.shuffle(all_data)
        
        # Split into train/validation sets (90/10 split)
        split_idx = int(0.9 * len(all_data))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Save in different formats
        self.save_conversation_format(train_data, val_data)
        self.save_alpaca_format(train_data, val_data)
        self.save_raw_text(train_data, val_data)
        
        # Generate statistics
        self.generate_statistics(all_data)
        
        return train_data, val_data
    
    def save_conversation_format(self, train_data: List[Dict], val_data: List[Dict]):
        """Save data in conversation format for training"""
        print("Saving conversation format...")
        
        train_conversations = [self.create_conversation_format(d) for d in train_data]
        val_conversations = [self.create_conversation_format(d) for d in val_data]
        
        with open('train_conversations.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_conversations))
        
        with open('val_conversations.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_conversations))
    
    def save_alpaca_format(self, train_data: List[Dict], val_data: List[Dict]):
        """Save data in Alpaca JSON format"""
        print("Saving Alpaca format...")
        
        train_alpaca = [self.create_alpaca_format(d) for d in train_data]
        val_alpaca = [self.create_alpaca_format(d) for d in val_data]
        
        with open('train_alpaca.json', 'w', encoding='utf-8') as f:
            json.dump(train_alpaca, f, ensure_ascii=False, indent=2)
        
        with open('val_alpaca.json', 'w', encoding='utf-8') as f:
            json.dump(val_alpaca, f, ensure_ascii=False, indent=2)
    
    def save_raw_text(self, train_data: List[Dict], val_data: List[Dict]):
        """Save raw text for tokenizer training"""
        print("Saving raw text...")
        
        all_text = []
        for data in train_data + val_data:
            all_text.append(data['instruction'])
            all_text.append(data['input'])
            all_text.append(data['output'])
        
        with open('sinhala_corpus.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
    
    def generate_statistics(self, all_data: List[Dict]):
        """Generate dataset statistics"""
        print("\nDataset Statistics:")
        print("=" * 50)
        
        type_counts = {}
        total_chars = 0
        total_words = 0
        
        for data in all_data:
            data_type = data['type']
            type_counts[data_type] = type_counts.get(data_type, 0) + 1
            
            text = data['instruction'] + " " + data['input'] + " " + data['output']
            total_chars += len(text)
            total_words += len(text.split())
        
        print(f"Total samples: {len(all_data)}")
        print(f"Total characters: {total_chars:,}")
        print(f"Total words: {total_words:,}")
        print(f"Average characters per sample: {total_chars/len(all_data):.1f}")
        print(f"Average words per sample: {total_words/len(all_data):.1f}")
        print("\nSample distribution:")
        for data_type, count in type_counts.items():
            print(f"  {data_type}: {count:,} ({count/len(all_data)*100:.1f}%)")

if __name__ == "__main__":
    processor = SinhalaDataProcessor()
    train_data, val_data = processor.prepare_training_data()
    print("\nData preparation completed successfully!") 