import { Pinecone } from "@pinecone-database/pinecone";
import "dotenv/config";

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const indexName = "portfolio";

const reset = async () => {
  try {
      console.log(`Checking for index: ${indexName}...`);
      const existingIndexes = await pc.listIndexes();
      
      const exists = existingIndexes.indexes && existingIndexes.indexes.some(idx => idx.name === indexName);

      if (exists) {
          console.log(`Deleting existing index "${indexName}" to clear old data...`);
          await pc.deleteIndex(indexName);
          console.log("Index deleted successfully.");
          console.log("Please wait about 30-60 seconds before running 'node db/loadDb.js' to allow Pinecone to reset.");
      } else {
          console.log(`Index "${indexName}" not found. You can proceed to load data directly.`);
      }
  } catch (error) {
      console.log("Error during reset:", error);
  }
}

reset();