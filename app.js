const TSNE = require('tsne-js')
const { OpenAIEmbeddings } = require("langchain/embeddings/openai")
const dotenv = require('dotenv')
const fs = require('fs');
const csv = require('csv-parser');
dotenv.config()



const csvFilePath = 'Steiner Search Terms.csv';  // Replace with the actual path to your CSV file
const data = [];

fs.createReadStream(csvFilePath)
  .pipe(csv())
  .on('data', (row) => {
    if (!data.includes(row['Search term'])) {
        data.push(row['Search term']);
    }
  })
  .on('end', () => {
    main(data)
  });


async function embed(arr) {

    console.log("Embedding...")

    const embeddings = new OpenAIEmbeddings({
        timeout: 10000, // 10s timeout
        verbose: true,
        openAIApiKey: process.env.OPENAI_API_KEY
      });
      /* Embed queries */
      const documentRes = await embeddings.embedDocuments(arr);
      
      return({ documentRes });
}


async function main(arr) {

    let {documentRes} = await embed(arr)

    let model = new TSNE({
        dim: 2,
        perplexity: 50.0,
        earlyExaggeration: 8.0,
        learningRate: 100.0,
        nIter: 10000,
        metric: 'euclidean'
        });
        
        // inputData is a nested array which can be converted into an ndarray
        // alternatively, it can be an array of coordinates (second argument should be specified as 'sparse')
        console.log("Initilaizing model...")
        model.init({
            data: documentRes,
            type: 'dense'
        });
        
        // `error`,  `iter`: final error and iteration number
        // note: computation-heavy action happens here

        console.log("Running model...")
        let [error, iter] = model.run();
        
        // // rerun without re-calculating pairwise distances, etc.
        // console.log("Re-running model...")
        // let [error2, iter2] = model.rerun();
        
        // `output` is unpacked ndarray (regular nested javascript array)
        console.log("Getting output")
        let output = model.getOutput();
        
        // `outputScaled` is `output` scaled to a range of [-1, 1]
        console.log("Getting scaled")
        let outputScaled = model.getOutputScaled();

        console.log(iter)
        console.log(error)
        console.log(outputScaled)

        const jsonFilePath = './output.json'



        try {
            fs.promises.writeFile(jsonFilePath, JSON.stringify(outputScaled, null, 2)).then(console.log('JSON file has been written successfully.'));
        } catch (error) {
            console.error('Error:', error);
        }

}

