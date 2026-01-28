# AI Chatbot Portfolio Component

A Next.js-based AI chatbot application designed to serve as an interactive component for **Pradeep M's Portfolio**. This project leverages modern AI technologies to provide intelligent conversational capabilities.

> **Note:** This project is intended to be linked and integrated with my personal portfolio website.

## 🚀 Tech Stack

- **Framework:** [Next.js](https://nextjs.org/) (App Router)
- **Styling:** [Tailwind CSS](https://tailwindcss.com/)
- **AI Integration:** [Vercel AI SDK](https://sdk.vercel.ai/docs)
- **LLM:** [Google Generative AI (Gemini)](https://ai.google.dev/)
- **Vector Database:** [Pinecone](https://www.pinecone.io/)
- **Orchestration:** [LangChain](https://js.langchain.com/)

## 🛠️ Getting Started

Follow these steps to run the project locally:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd chat-bot
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Environment Setup:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    ```

4.  **Run the development server:**
    ```bash
    npm run dev
    ```

    Open [http://localhost:3000](http://localhost:3000) (or the port shown in your terminal) to view the application.

## 📦 Deployment

This application is optimized for deployment on **Vercel**.

1.  Push your code to a Git repository (GitHub, GitLab, etc.).
2.  Import the project into Vercel.
3.  Add the `GOOGLE_API_KEY` and `PINECONE_API_KEY` in the Vercel project settings (Environment Variables).
4.  Deploy!

## 📄 License

This project is for personal portfolio usage.
