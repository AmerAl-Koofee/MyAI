# Chatbot Project

## 📌 Overview

This is a Django-based chatbot project that uses **Hugging Face models** and **pgvector** for embedding storage. It allows users to upload PDFs, process embeddings, and interact with a chatbot.

## 🚀 Features

- **Upload PDFs** for processing
- **Delete PDFs** and associated embeddings
- **Chat with the AI model** (Meta-Llama 3.2-3B-Instruct)
- **Uses PostgreSQL + pgvector for embeddings storage**
- **Runs on CPU (no GPU required)**

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up `.env` File

Create a file at the project directory and name it `.env`. Open the file and add the following:

```
POSTGRES_DATABASE=mydb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword
DATABASE_URL=postgresql+psycopg2://postgres:yourpassword@localhost:5432/mydb
HUGGINGFACE_TOKEN=your_token_here
```

### 5️⃣ Run Migrations & Start Server

```bash
python manage.py migrate
python manage.py runserver
```

Your Django server should now be running at **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** 🎉

---

## 🛄 Database Setup (PostgreSQL + pgvector)

### 1. Start PostgreSQL (Docker recommended)
```bash
docker run --name pgvector-db -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=yourpassword -e POSTGRES_DB=mydb -d pgvector/pgvector
```

### 2. Connect to the database
```bash
docker exec -it pgvector-db psql -U postgres -d mydb
```

### 3. Create the `pgvector` extension inside PostgreSQL
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## 🤖 Hugging Face Authentication

To access **Meta-Llama 3.2-3B-Instruct**, log in to Hugging Face:

```bash
huggingface-cli login
```

---

## 🎯 API Endpoints

| Method   | Endpoint            | Description                     |
| -------- | ------------------- | ------------------------------- |
| `GET`    | `/api/list/`        | List uploaded PDFs              |
| `POST`   | `/api/upload/`      | Upload a PDF                    |
| `DELETE` | `/api/delete/{id}/` | Delete a PDF and its embeddings |

---

