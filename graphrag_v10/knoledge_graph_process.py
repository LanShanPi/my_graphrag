import os
import re
import networkx as nx
from pyvis.network import Network
import rdflib
from rdflib import Namespace, Graph
import logging
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import PropertyGraphIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
import owlready2
from owlready2 import get_ontology, sync_reasoner

# 如果用 Owlready2
try:
    import owlready2
    HAVE_OWLREADY2 = True
except ImportError:
    HAVE_OWLREADY2 = False

from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from prompt.prompt import highschool_physics as triplet_extraction_template

def get_config_prompt():
    # 处理图谱生成模版
    CUSTOM_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
        triplet_extraction_template["CUSTOM_KG_TRIPLET_EXTRACT_TMPL"],
        prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
    )
    allowed_entity_types = triplet_extraction_template["allowed_entity_types"]
    allowed_relation_types = triplet_extraction_template["allowed_relation_types"]
    return allowed_entity_types, allowed_relation_types, CUSTOM_KG_TRIPLET_EXTRACT_PROMPT

def store_in_rdf(index, storage_dir, config):
    """
    将 LlamaIndex 的三元组保存为 knowledge_graph.ttl (RDF/Turtle)，
    以便后续做 SPARQL 或 OWL 推理。
    """
    g = rdflib.Graph()
    EX = Namespace("http://example.org/")

    for triplet in index.property_graph_store.get_triplets():
        subj = triplet["head"]
        rel = triplet["relation"]
        obj = triplet["tail"]

        def to_uri_fragment(text):
            return re.sub(r'[^a-zA-Z0-9_]+', '_', text.strip())

        subj_uri = EX[to_uri_fragment(subj)]
        rel_uri  = EX[to_uri_fragment(rel)]
        obj_uri  = EX[to_uri_fragment(obj)]
        g.add((subj_uri, rel_uri, obj_uri))

    rdf_file = os.path.join(storage_dir, "knowledge_graph.ttl")
    g.serialize(destination=rdf_file, format="turtle")
    if config.debug:
        logging.info(f"[DEBUG] RDF knowledge graph stored in: {rdf_file}")

def advanced_symbolic_reasoning_owlready2(storage_dir, config):
    """
    演示: 用 Owlready2 加载 knowledge_graph.ttl 并执行推理(SWRL等)。
    """
    if not HAVE_OWLREADY2:
        logging.warning("[Owlready2] Not installed. Skip advanced reasoning.")
        return

    rdf_file = os.path.join(storage_dir, "knowledge_graph.ttl")
    if not os.path.exists(rdf_file):
        logging.warning(f"[Owlready2] {rdf_file} not found, cannot do OWL reasoning.")
        return

    world = owlready2.World()
    # 加载三元组到本体
    world.get_ontology(f"file://{rdf_file}").load()

    physics_onto = world.get_ontology("http://example.org/physics_onto#")

    with physics_onto:
        # 定义物理类和属性
        class PhysicalLaw(Thing): pass
        class Formula(Thing): pass
        class Experiment(Thing): pass

        class hasFormula(ObjectProperty):
            domain = [PhysicalLaw]
            range = [Formula]

        class testedBy(ObjectProperty):
            domain = [PhysicalLaw]
            range = [Experiment]

        # 定义一个简单规则（假设）
        rule = Imp()
        rule.set_as_rule("PhysicalLaw(?l) ^ hasFormula(?l, ?f) -> Formula(?f)")

    try:
        sync_reasoner()
        logging.info("[OWLready2] Reasoning completed.")
    except Exception as e:
        logging.error("[OWLready2] sync_reasoner() failed:", e)

def generate_knowledge_graph(documents, dir_name, storage_dir, config):
    """
    从分块后的 Document 列表构建知识图谱，保存为 LlamaIndex + RDF + PyVis可视化
    """
    llm = LlamaOpenAI(temperature=0, model="gpt-4o", api_key="", base_url="", timeout=600)
    embed_model = OpenAIEmbedding(model_name="text-embedding-ada-002", api_key="", base_url="")

    entity_types, relation_types, CUSTOM_KG_TRIPLET_EXTRACT_PROMPT = get_config_prompt()

    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = PropertyGraphIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model,
        include_embeddings=False,
        max_triplets_per_chunk=config.max_triplets_per_chunk,
        storage_context=storage_context,
        show_progress=True,
        kg_triple_extract_template=CUSTOM_KG_TRIPLET_EXTRACT_PROMPT,
        allowed_entity_types=entity_types,
        allowed_relation_types=relation_types,
    )

    index_dir = os.path.join(storage_dir, "index")
    os.makedirs(index_dir, exist_ok=True)
    storage_context.persist(persist_dir=index_dir)

    G = nx.DiGraph()
    for triplet in index.property_graph_store.get_triplets():
        subj = triplet["head"]
        obj = triplet["tail"]
        rel = triplet["relation"]
        time = triplet.get("time", "未知时间")
        G.add_node(subj, label=subj)
        G.add_node(obj, label=obj)
        G.add_edge(subj, obj, label=f"{rel} ({time})")

    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(G)

    html_file = os.path.join(storage_dir, f"{dir_name}_graph.html")
    net.show(html_file)
    logging.info(f"HTML file has been generated: {html_file}")

    store_in_rdf(index, storage_dir, config)

    if config.enable_symbolic_reasoning and config.reasoning_mode == "owl":
        advanced_symbolic_reasoning_owlready2(storage_dir, config)

    return index

def load_knowledge_graph(storage_dir):
    """
    从本地加载已经构建好的 LlamaIndex 索引
    """
    storage_context = StorageContext.from_defaults(persist_dir=os.path.join(storage_dir, "index"))
    index = load_index_from_storage(storage_context)
    logging.info("[INFO] Loaded existing knowledge graph from disk.")
    return index

def symbolic_query_sparql(storage_dir, limit=5):
    """
    演示如何用 SPARQL 查询 knowledge_graph.ttl
    """
    from rdflib.plugins.sparql import prepareQuery
    rdf_file = os.path.join(storage_dir, "knowledge_graph.ttl")
    if not os.path.exists(rdf_file):
        return []

    g = rdflib.Graph()
    g.parse(rdf_file, format="turtle")

    query_str = f"""
    SELECT ?s ?p ?o
    WHERE {{
      ?s ?p ?o .
    }} LIMIT {limit}
    """
    q = prepareQuery(query_str)
    res = g.query(q)
    results = [f"{str(row.s)}, {str(row.p)}, {str(row.o)}" for row in res]
    return results
