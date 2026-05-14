# Generic RAG framework skeleton — Layer 2.
# ``framework.nodes.pipeline`` pulls default prompt/prescript text from ``alkawarzmi`` for
# module-level defaults; ``build_graph`` always runs ``configure_pipeline`` so production
# uses the injected :class:`framework.profile.RAGProfile`.
