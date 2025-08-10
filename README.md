# Configuration: .env File

To use this project, you need to configure a `.env` file in the project root with your Azure OpenAI credentials. Create a file named `.env` and add the following variables:


```
API_KEY="your-key"
ENDPOINT="https://<your-resource>.openai.azure.com/"
API_VERSION="2024-12-01-preview"
DEPLOYMENT_NAME="your-deployment-name"
MODEL="azure_openai:gpt-5-mini"
```



Replace `your-key`, `<your-resource>`, `your-deployment-name`, and `azure_openai:gpt-5-mini` with your actual API key, resource name, deployment name, and model string.

The `.env` file is loaded automatically by the application and should not be committed to version control (it is already included in `.gitignore`).