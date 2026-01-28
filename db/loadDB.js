import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import "dotenv/config";
import sampleData from "./sample-data.json" with { type: "json" };

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-embedding-001" });

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const indexName = "portfolio";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const createCollection = async () => {
  try {
      const existingIndexes = await pc.listIndexes();
      if (existingIndexes.indexes && existingIndexes.indexes.some(idx => idx.name === indexName)) {
         console.log("Index already exists");
         return;
      }

      await pc.createIndex({
          name: indexName,
          dimension: 3072, 
          metric: 'cosine',
          spec: {
              serverless: {
                  cloud: 'aws',
                  region: 'us-east-1' 
              }
          }
      });
      console.log("Index created. Waiting for initialization...");
      await new Promise(resolve => setTimeout(resolve, 20000)); 
  } catch (error) {
      console.log("Error checking/creating index:", error);
  }
}

const loadData = async () => {
    const index = pc.index(indexName);
    
    for await (const { id, info, description } of sampleData) {
        let textChunks = [];

        // Handle different data types to create meaningful text chunks
        if (Array.isArray(description)) {
            // For arrays (like projects or jobs), treat each item as a separate chunk if complex,
            // or join them if simple strings.
            const isRefArray = description.every(item => typeof item === 'object');
            
            if (isRefArray) {
                // Array of objects (e.g. projects) -> convert each to a string
                textChunks = description.map(item => JSON.stringify(item));
            } else {
                // Array of strings -> join them with newlines
                textChunks = [description.join('\n')];
            }
        } else if (typeof description === 'object') {
            // Single object -> valid JSON string
            textChunks = [JSON.stringify(description)];
        } else {
            // String -> use as is
            textChunks = [description];
        }

        // Now split/process each base chunk (in case one single project is huge)
        let i = 0;
        for (const text of textChunks) {
            const splitChunks = await splitter.splitText(text);
            
            for (const chunk of splitChunks) {
                const result = await model.embedContent(chunk);
                const embedding = result.embedding.values;

                await index.upsert([
                    {
                        id: `${id}_${i}`,
                        values: embedding,
                        metadata: {
                            document_id: id,
                            info: info,
                            // Store the chunk text so the model can read it
                            description: chunk 
                        }
                    }
                ]);
                i++;
            }
        }
    }

    console.log("data added");
}

createCollection().then(() => loadData());
