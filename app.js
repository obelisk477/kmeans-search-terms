const TSNE = require('tsne-js')
const { OpenAIEmbeddings } = require("langchain/embeddings/openai")
const dotenv = require('dotenv')
dotenv.config()


const array = ['hello world', 'goodbye world', 'see you later world']

async function embed(arr) {
    const embeddings = new OpenAIEmbeddings({
        timeout: 10000, // 1s timeout
        verbose: true,
        openAIApiKey: process.env.OPENAI_API_KEY
      });
      /* Embed queries */
      const documentRes = await embeddings.embedDocuments(array);
      
      return({ documentRes });
}


async function main(arr) {

    let {documentRes} = await embed(arr)

    console.log()
    console.log("hello")

    let model = new TSNE({
        dim: 2,
        perplexity: 30.0,
        earlyExaggeration: 4.0,
        learningRate: 100.0,
        nIter: 1000,
        metric: 'euclidean'
        });
        
        // inputData is a nested array which can be converted into an ndarray
        // alternatively, it can be an array of coordinates (second argument should be specified as 'sparse')
        model.init({
            data: documentRes,
            type: 'dense'
        });
        
        // `error`,  `iter`: final error and iteration number
        // note: computation-heavy action happens here
        // let [error, iter] = model.run();
        
        // rerun without re-calculating pairwise distances, etc.
        // let [error, iter] = model.rerun();
        
        // `output` is unpacked ndarray (regular nested javascript array)
        let output = model.getOutput();
        
        console.log(output)
        
        // `outputScaled` is `output` scaled to a range of [-1, 1]
        let outputScaled = model.getOutputScaled();
        
        console.log(outputScaled)
}

main()