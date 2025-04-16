                           +------------------+
                           |  User Query      |
                           +--------+---------+
                                    |
                                    v
                    +-----------------------------+
                    |   SemanticRetriever          |
                    |-----------------------------|
                    | 1. Vector Search (FAISS)     |-----> Embedding Model (OpenAI)
                    | 2. Ontology Context Lookup   |-----> OntologyManager
                    | 3. Semantic Path Discovery   |-----> Ontology Graph
                    +-----------------------------+
                                    |
                +-------------------+--------------------+
                |                                        |
        +---------------+                       +----------------+
        | FAISS Results  |                       | Ontology Paths |
        +---------------+                       +----------------+
                |                                        |
                +--------------------+------------------+
                                     |
                          +-----------------------+
                          |  Retrieved Documents  |
                          +-----------------------+
