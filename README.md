# How to Run

To setup everything, format and lint (you should do this often and in case of git hook failures):
```bash
make all
```

To run the simple agent example:

```bash
make run-simple-agent
```

To run the mcps example:

```bash
make run-mcps
```
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



# Example Interaction

```
======================================== New Event ========================================
Tool call: get_lat_long with args: {'city': 'Zurich'}

======================================== New Event ========================================
Tool response: {"latitude":"47.3744489","longitude":"8.5410422"}

======================================== New Event ========================================
Tool call: temperature with args: {'latitude': 47.3744489, 'longitude': 8.5410422}

======================================== New Event ========================================
Tool response: {"time":["2025-08-10T00:00","2025-08-10T01:00","2025-08-10T02:00","2025-08-10T03:00","2025-08-10T04:00","2025-08-10T05:00","2025-08-10T06:00","2025-08-10T07:00","2025-08-10T08:00","2025-08-10T09:00","2025-08-10T10:00","2025-08-10T11:00","2025-08-10T12:00","2025-08-10T13:00","2025-08-10T14:00","2025-08-10T15:00","2025-08-10T16:00","2025-08-10T17:00","2025-08-10T18:00","2025-08-10T19:00","2025-08-10T20:00","2025-08-10T21:00","2025-08-10T22:00","2025-08-10T23:00"],"temperature_2m":[24.0,23.5,22.6,21.4,20.7,20.4,20.8,22.6,24.1,25.9,27.2,28.4,29.0,29.7,30.3,30.3,30.1,29.6,28.8,26.7,24.7,23.5,22.7,21.8]}

======================================== New Event ========================================
Tool call: current_date_time with args: {}

======================================== New Event ========================================
Tool response: {"date":"2025-08-10","time":"19:20:27"}

======================================== New Event ========================================
Message content: Zurich now (local time 2025-08-10 19:20): about 26.7°C (latest hourly data at 19:00).

Evening forecast:
- 20:00: ~24.7°C
- 21:00: ~23.5°C
- 22:00: ~22.7°C
- 23:00: ~21.8°C

Want this in Fahrenheit or a more detailed forecast for the next 24 hours?
```