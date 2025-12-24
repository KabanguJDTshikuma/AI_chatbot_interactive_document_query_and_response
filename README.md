# Build a Chatbot for Your Data

## Introduction

Create a chatbot for your own PDF file using Flask, a popular web framework, and LangChain, another popular framework for working with large language models (LLMs). The chatbot you develop will not just interact with users through text but also comprehend and answer questions related to the content of a specific document.

At the end of this project, you will gain a deeper understanding of chatbots, web application development using Flask and Python, and the use of LangChain framework in interpreting and responding to a wide array of user inputs. And most important, you would have built a comprehensive and impressive chatbot application!

## Key Elements

### Chatbots
Chatbots are software applications designed to engage in human-like conversation. They can respond to text inputs from users and are widely used in various domains, including customer service, eCommerce, and education. In this project, you will build a chatbot capable of not only engaging users in a general conversation but also answering queries based on a particular document.

### LangChain
LangChain is a versatile tool for building AI-driven language applications. It provides various functionalities such as text retrieval, summarization, translation, and many more, by leveraging pretrained language models. In this project, you will be integrating LangChain into your chatbot, empowering it to understand and respond to diverse user inputs effectively.

### Flask
Flask is a lightweight and flexible web framework for Python, known for its simplicity and speed. A web framework is a software framework designed to support the development of web applications, including the creation of web servers, and management of HTTP requests and responses.

You will use Flask to create the server side or backend of your chatbot. This involves handling incoming messages from users, processing these messages, and sending appropriate responses back to the user.

### Routes in Flask
Routes are an essential part of web development. When your application receives a request from a client (typically a web browser), it needs to know how to handle that request. This is where routing comes in.

In Flask, routes are created using the `@app.route` decorator to bind a function to a URL route. When a user visits that URL, the associated function is executed. In your chatbot project, you will use routes to handle the POST requests carrying user messages and to process document data.

### HTML - CSS - JavaScript
You are provided with a ready-to-use chatbot front-end, built with HTML, CSS, and JavaScript. HTML structures web content, CSS styles it and JavaScript adds interactivity. These technologies create a visually appealing and interactive chatbot interface.

## Learning Objectives

At the end of this project, you will be able to:
* Explain the basics of Langchain and AI applications
* Set up a development environment for building an assistant using Python Flask
* Implement PDF upload functionality to allow the assistant to comprehend file input from users
* Integrate the assistant with open source models to give it a high level of intelligence and the ability to understand and respond to user requests
* (Optional) Deploy the PDF assistant to a web server for use by a wider audience

## Prerequisites

Knowledge of the basics of HTML/CSS, JavaScript, and Python is nice to have but not essential. Each step of the process and code will have a comprehensive explanation in this lab.

## Getting Started

### Setting up the Environment

First, let's set up the environment by executing the following code:

```bash
pip3 install virtualenv 
virtualenv my_env # create a virtual environment my_env
source my_env/bin/activate # activate my_env
```

### Installation

Run the following commands to retrieve the project, give it an appropriate name, and finally move to that directory:

```bash
git clone https://github.com/ibm-developer-skills-network/wbphl-build_own_chatbot_without_open_ai.git
mv wbphl-build_own_chatbot_without_open_ai build_chatbot_for_your_data
cd build_chatbot_for_your_data
```

Install the requirements for the project:

```bash
pip install -r requirements.txt
pip install langchain-community
```

## Project Structure

### Frontend (HTML, CSS, and JavaScript)
* `index.html`: Responsible for the layout and structure of the web interface.
* `style.css`: Responsible for customizing the visual appearance of the page's components.
* `script.js`: Responsible for the page's interactivity and functionality.

### Backend (Worker)
`worker.py` is part of a chatbot application that processes user messages and documents. It uses the langchain library.

**Key Functions:**
* `init_llm()`: Initializes the language model and embeddings.
* `process_document(document_path)`: Processes a given PDF document (loading, splitting, creating embeddings).
* `process_prompt(prompt)`: Processes a user's prompt or question using the retrieval chain.

### Backend (Server)
`server.py` handles the web server operations using Flask.

## Running the App

To implement your chatbot, you need to run the `server.py` file first.

```bash
python3 server.py
```

If you are running the file locally, you can open the browser and go to `http://127.0.0.1:8000`.
