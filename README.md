> [!Important]
> This is an experimental repository & is not meant to be production ready!

### GOAL

The goal was to create a 2-billion-parameter language model capable of running on a laptop. One issue with products like ChatGPT and Claude is that they don’t understand government services—like Singpass, CPF, or the National Service Portal—well enough for Singaporeans, especially senior citizens who need the most help. As a result, they can’t effectively assist this population with their day to day tasks

<br>

### How the model & framework was created

- **Training & Model:**
  - **Frameworks:** PyTorch, Hugging Face Transformers
  - **Hardware:** AWS EC2 p4d.24xlarge instance
  - **AMI:** Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04)
- **API Layer:**
  - **Language:** Golang
  - **Framework:** Gin
  - **Communication:** REST API (with error handling and HTTP status codes)
- **Inference Service:**
  - **Framework:** FastAPI served with Uvicorn
  - **Integration:** Direct HTTP calls from the Go API
- **Data & Storage:**
  - **Data Sources:**
    1. [Common Crawl](https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-22/index.html)
    2. [ChatGPT Conversations](https://www.kaggle.com/datasets/noahpersaud/89k-chatgpt-conversations)
    3. [Wikipedia](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701)
    4. [Reddit Comments](https://github.com/PolyAI-LDN/conversational-datasets/tree/master/reddit)
    5. [BookCorpus](https://paperswithcode.com/dataset/bookcorpus)
    6. [Project Gutenberg](https://www.gutenberg.org)
    7. Government websites scraped via Playwright

### MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
