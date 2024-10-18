python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/Products \
  --index data/indexes/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw