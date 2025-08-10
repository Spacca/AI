# Configuration: .env File

To use this project, you need to configure a `.env` file in the project root with your Azure OpenAI credentials. Create a file named `.env` and add the following variables:

```
AZURE_OPENAI_API_KEY="your-key"
AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name"
```


Replace `your-key`, `<your-resource>`, and `your-deployment-name` with your actual Azure OpenAI API key, resource name, and deployment name.

The `.env` file is loaded automatically by the application and should not be committed to version control (it is already included in `.gitignore`).