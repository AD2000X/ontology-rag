# This completes the app.py file from previous section
    query = st.text_input(
        "Enter a question to compare RAG approaches:",
        "How does customer feedback influence product development?"
    )
    
    if query:
        col1, col2 = st.columns(2)
        
        with st.spinner("Running both RAG approaches..."):
            # Traditional RAG - using just vector similarity
            with col1:
                st.subheader("Traditional RAG")
                
                # Retrieve context using basic vector search
                vector_docs = semantic_retriever.vector_store.similarity_search(query, k=k_val)
                
                # Construct system message with retrieved context
                vector_context = "\n\n".join([doc.page_content for doc in vector_docs])
                
                vector_system_message = f"""You are an enterprise knowledge assistant.
                Answer the user's question based on the following retrieved context. 
                If the information isn't in the context, acknowledge that you don't know.
                
                Context:
                {vector_context}
                """
                
                # Generate response from LLM
                vector_messages = [
                    {"role": "system", "content": vector_system_message},
                    {"role": "user", "content": query}
                ]
                
                vector_response = llm.invoke(vector_messages)
                vector_answer = vector_response.content
                
                # Display the answer
                st.markdown("#### Answer")
                st.write(vector_answer)
                
                # Show retrieved context
                st.markdown("#### Retrieved Context")
                for i, doc in enumerate(vector_docs):
                    with st.expander(f"Source {i+1}"):
                        st.code(doc.page_content)
            
            # Ontology-Enhanced RAG
            with col2:
                st.subheader("Ontology-Enhanced RAG")
                
                # Retrieve context using the semantic retriever with ontology awareness
                retrieval_result = semantic_retriever.retrieve_with_paths(
                    query, k=k_val
                )
                
                retrieved_docs = retrieval_result["documents"]
                
                # Construct system message with retrieved context
                enhanced_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                enhanced_system_message = f"""You are an enterprise knowledge assistant with access to ontology-structured information.
                Answer the user's question based on the following retrieved context. 
                If the information isn't in the context, acknowledge that you don't know.
                
                Context:
                {enhanced_context}
                """
                
                # Generate response from LLM
                enhanced_messages = [
                    {"role": "system", "content": enhanced_system_message},
                    {"role": "user", "content": query}
                ]
                
                enhanced_response = llm.invoke(enhanced_messages)
                enhanced_answer = enhanced_response.content
                
                # Display the answer
                st.markdown("#### Answer")
                st.write(enhanced_answer)
                
                # Show retrieved context
                st.markdown("#### Retrieved Context")
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get("source", "unknown")
                    
                    if source == "ontology":
                        with st.expander(f"Ontology Context {i+1}"):
                            st.code(doc.page_content)
                    elif source == "text":
                        with st.expander(f"Text Context {i+1}"):
                            st.code(doc.page_content)
                    elif source == "ontology_context":
                        with st.expander(f"Semantic Context {i+1}"):
                            st.markdown(doc.page_content)
                    elif source == "semantic_path":
                        with st.expander(f"Relationship Path {i+1}"):
                            st.markdown(doc.page_content)
                    else:
                        with st.expander(f"Source {i+1}"):
                            st.code(doc.page_content)
        
        # Analysis of differences
        st.markdown("---")
        st.subheader("💡 Analysis of Differences")
        
        st.markdown("""
        The comparison above demonstrates several key advantages of ontology-enhanced RAG:
        
        1. **Structural Awareness**: The ontology-enhanced approach understands the relationships between entities, not just their textual similarity.
        
        2. **Multi-hop Reasoning**: By using the knowledge graph structure, the enhanced approach can connect information across multiple relationship hops.
        
        3. **Context Enrichment**: The ontology provides additional context about entity types, properties, and relationships that isn't explicit in the text.
        
        4. **Inference Capabilities**: The structured knowledge allows for logical inferences that vector similarity alone cannot achieve.
        
        Try more complex queries that require understanding relationships to see the differences more clearly!
        """)