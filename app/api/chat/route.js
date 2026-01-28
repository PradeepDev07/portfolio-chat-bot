import { GoogleGenerativeAI } from "@google/generative-ai";
import { GoogleGenerativeAIStream, StreamingTextResponse } from "ai";
import { Pinecone } from "@pinecone-database/pinecone";

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

async function retry(fn, retries = 3, delay = 2000) {
  try {
    return await fn();
  } catch (error) {
    if (retries > 0 && (error.status === 429 || error.status === 503)) {
      console.log(`Rate limit hit. Retrying in ${delay}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      return retry(fn, retries - 1, delay * 2);
    }
    throw error;
  }
}

export async function POST(req) {
  try {
    const { messages } = await req.json();

    const latestMessage = messages[messages?.length - 1]?.content;
    let docContext = "";

    const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-001" });
    const result = await retry(() => embeddingModel.embedContent(latestMessage));
    const embedding = result.embedding.values;

    const index = pc.index("portfolio");

    const queryResponse = await index.query({
      vector: embedding,
      topK: 5,
      includeMetadata: true,
    });

    docContext = `
          START CONTEXT
          ${queryResponse.matches.map((match) => match.metadata?.description).join("\n")}
          END CONTEXT
          `;

    const systemInstruction = `
              You are an AI assistant answering questions as Pradeep M in his Portfolio App. 
              Format responses using markdown where applicable.
              
              Context:
              ${docContext}
              
              If the answer is not provided in the context, say:
              "I'm sorry, I do not know the answer".
              `;

    const model = genAI.getGenerativeModel({ 
        model: "gemini-2.5-flash",
        systemInstruction: systemInstruction 
    });

    // Filter out system messages if any, and map to Gemini format
    const geminiMessages = messages
        .filter(m => m.role !== 'system')
        .map(m => ({
            role: m.role === 'user' ? 'user' : 'model',
            parts: [{ text: m.content }]
        }));

    const geminiStream = await retry(() => model.generateContentStream({
      contents: geminiMessages,
    }));

    const stream = GoogleGenerativeAIStream(geminiStream);
    return new StreamingTextResponse(stream);
  } catch (e) {
    throw e;
  }
}
