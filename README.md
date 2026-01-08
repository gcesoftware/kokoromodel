# Kokoro TTS AI Assets Provider

This repository serves as a remote asset provider for the **[您的 App 名称]** Windows application. It hosts necessary AI model weights and voice assets to comply with Microsoft Store packaging size recommendations.

## 📦 Hosted Assets

| Asset Name | Version | Description |
| :--- | :--- | :--- |
| `kokoro-v1.1-zh.pth` | 1.1 | Multilingual (CN/EN) model weights for Kokoro TTS. |
| `Voices/` | 1.0 | Voice profile characteristics (.pt files). |

## ⚖️ License & Attribution

The model weights hosted in this repository are derived from the **Kokoro-82M** project by **hexgrad**.

- **Original Repository**: [https://huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- **License**: Apache License 2.0

According to the Apache 2.0 License, you are free to use, modify, and distribute these files as long as the original license and copyright notices are preserved.

## 🛠 Usage in App

This repository is accessed via direct download links from a C# desktop application. Files are downloaded to the user's `LocalState` directory to ensure optimal performance and persistent storage.
