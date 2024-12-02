mingchaonaxieshi = {
    "CUSTOM_KG_TRIPLET_EXTRACT_TMPL":(
            "从给定文本中提取与明朝历史相关的知识三元组。"
            "每个三元组应以 (head, relation, tail) 及其各自的类型形式出现。\n"
            "---------------------\n"
            "初始本体论：\n"
            "实体类型：{allowed_entity_types}\n"
            "关系类型：{allowed_relation_types}\n"
            "\n"
            "以这些类型为起点，但根据上下文需要引入新类型。\n"
            "\n"
            "指导原则：\n"
            "- 以 JSON 格式输出：[{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
            "- 使用最完整的实体名称（例如'朱元璋'而不是'元璋'）\n"
            "- 保持实体简洁（最多 3-5 个词）\n"
            "- 对复杂事件分解为多个三元组，例如：\n"
            "  靖难之役导致了明成祖的登基和明初中央集权的加强 =>\n"
            "  [{{'head': '靖难之役', 'relation': 'RESULTED_IN', 'tail': '明成祖登基', 'tail_type': 'EVENT'}},\n"
            "   {{'head': '靖难之役', 'relation': 'RESULTED_IN', 'tail': '中央集权加强', 'tail_type': 'SOCIAL_SYSTEM'}}]\n"
            "- 特别关注政策的背景、执行及其影响（如'张居正改革'）\n"
            "- 提取事件间的因果关系（如'土木堡之变导致明英宗被俘虏'）\n"
            "---------------------\n"
            "文本：{text}\n"
            "输出：\n"
        ),
    "allowed_entity_types":[
        "PERSON", "EVENT", "POLITICAL_FIGURE", "MILITARY_LEADER",
        "DYNASTY", "BATTLE", "CAPITAL", "POLICY", "INSTITUTION",
        "CULTURAL_FIGURE", "ECONOMIC_POLICY", "SOCIAL_SYSTEM"
    ],
    "allowed_relation_types":[
        "FOUNDER", "MEMBER", "GENERAL", "CAPTURED_BY", "SUCCEEDED_BY",
        "PARTICIPANT", "RESULTED_IN", "IMPLEMENTED", "INFLUENCED_BY",
        "BORN_IN", "DIED_IN", "ASCENDED_THRONE", "STARTED_REIGN",
        "ENDED_REIGN", "LED_BATTLE", "DEFEATED"
    ]

}


hongloumeng = {
    "CUSTOM_KG_TRIPLET_EXTRACT_TMPL":(
                "从给定的文本中提取多层次的知识三元组，"
                "包括直接关系、间接关系、情感联系、物品归属、地理位置、事件参与等，"
                "每个三元组以 (head, relation, tail) 及其各自的类型形式出现。\n"
                "---------------------\n"
                "初始本体论：\n"
                "实体类型：{allowed_entity_types}\n"
                "关系类型：{allowed_relation_types}\n"
                "指导原则：\n"
                "- 以 JSON 格式输出：[{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
                "- 使用最完整的实体形式（例如'贾宝玉'而不是'宝玉'）\n"
                "- 对复杂的关系进行分解，例如：\n"
                "  1. '林黛玉对贾宝玉既有爱情也有嫉妒' =>\n"
                "     [{{'head': '林黛玉', 'relation': 'EMOTION_TOWARDS', 'tail': '贾宝玉', 'tail_type': 'PERSON', 'emotion': '爱'}},\n"
                "      {{'head': '林黛玉', 'relation': 'EMOTION_TOWARDS', 'tail': '贾宝玉', 'tail_type': 'PERSON', 'emotion': '嫉妒'}}]\n"
                "- 保持知识图谱连贯，捕获上下文信息（例如家庭背景、事件背景）\n"
                "- 特别关注《红楼梦》中的隐喻（例如'通灵宝玉象征贾宝玉的命运'）\n"
                "- 提取《红楼梦》的多层次结构（人物、情感、社会关系、哲学思想）\n"
                "---------------------\n"
                "文本：{text}\n"
                "输出：\n"
            ),
    "allowed_entity_types":[
        "PERSON","FAMILY","PLACE","OBJECT","EVENT","EMOTION","IDEA"],
    "allowed_relation_types":[
        "RELATIVE_OF","FRIEND_OF","ENEMY_OF","MENTOR_OF","OWNER_OF","LOCATED_IN",
        "PART_OF","INVOLVED_IN","EMOTION_TOWARDS","SYMBOLIZES","IMPACTS","CAUSED_BY","BELONGS_TO",
    ]

}


zhuyuanzhang = {
    "CUSTOM_KG_TRIPLET_EXTRACT_TMPL": (
        "从给定文本中提取与指定名词 '朱元璋' 相关的知识三元组。\n"
        "仅保留那些与目标名词直接相关的三元组，目标名词可以出现在 head 或 tail 中。\n"
        "每个三元组应以 (head, relation, tail) 及其各自的类型形式出现。\n"
        "---------------------\n"
        "初始本体论：\n"
        "实体类型：{allowed_entity_types}\n"
        "关系类型：{allowed_relation_types}\n"
        "\n"
        "以这些类型为起点，但根据上下文需要引入新类型。\n"
        "\n"
        "指导原则：\n"
        "- 以 JSON 格式输出：[{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
        "- 使用最完整的实体名称（例如'朱元璋'而不是'元璋'）\n"
        "- 保持实体简洁（最多 3-5 个词）\n"
        "- 确保目标名词 '{target_entity}' 始终出现在 head 或 tail 中\n"
        "- 对复杂事件分解为多个三元组，例如：\n"
        "  靖难之役导致了明成祖的登基和明初中央集权的加强 =>\n"
        "  [{{'head': '靖难之役', 'relation': 'RESULTED_IN', 'tail': '明成祖登基', 'tail_type': 'EVENT'}},\n"
        "   {{'head': '靖难之役', 'relation': 'RESULTED_IN', 'tail': '中央集权加强', 'tail_type': 'SOCIAL_SYSTEM'}}]\n"
        "- 特别关注与目标名词 '{target_entity}' 的政策、事件及影响（如'朱元璋废除丞相制度'）\n"
        "- 提取事件间的因果关系（如'朱元璋建立明朝导致了中央集权的形成'）\n"
        "---------------------\n"
        "目标名词：{target_entity}\n"
        "文本：{text}\n"
        "输出：\n"
    ),
    "allowed_entity_types": [
        "PERSON", "EVENT", "POLITICAL_FIGURE", "MILITARY_LEADER",
        "DYNASTY", "BATTLE", "CAPITAL", "POLICY", "INSTITUTION",
        "CULTURAL_FIGURE", "ECONOMIC_POLICY", "SOCIAL_SYSTEM"
    ],
    "allowed_relation_types": [
        "FOUNDER", "MEMBER", "GENERAL", "CAPTURED_BY", "SUCCEEDED_BY",
        "PARTICIPANT", "RESULTED_IN", "IMPLEMENTED", "INFLUENCED_BY",
        "BORN_IN", "DIED_IN", "ASCENDED_THRONE", "STARTED_REIGN",
        "ENDED_REIGN", "LED_BATTLE", "DEFEATED"
    ]
}

# from llama_index.core.prompts import PromptTemplate
# from llama_index.core.prompts.prompt_type import PromptType

# # 定义自定义提示模板
# CUSTOM_NOUN_TRIPLET_EXTRACT_TMPL = (
#     "从给定文本中提取与名词 '{noun}' 相关的知识三元组。"
#     "每个三元组应以 (head, relation, tail) 及其各自的类型形式呈现。\n"
#     "---------------------\n"
#     "初始本体论:\n"
#     "实体类型: {allowed_entity_types}\n"
#     "关系类型: {allowed_relation_types}\n"
#     "\n"
#     "以这些类型为起点，但根据上下文需要引入新类型。\n"
#     "\n"
#     "指南:\n"
#     "- 以 JSON 格式输出: [{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
#     "- 使用实体的最完整形式（例如，使用 'United States of America' 而不是 'USA'）\n"
#     "- 保持实体简洁（最多 3-5 个词）\n"
#     "- 将复杂短语分解为多个三元组\n"
#     "- 确保知识图谱连贯且易于理解\n"
#     "---------------------\n"
#     "文本: {text}\n"
#     "输出:\n"
# )

# CUSTOM_NOUN_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
#     CUSTOM_NOUN_TRIPLET_EXTRACT_TMPL, 
#     prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
# )

# # 定义允许的实体类型和关系类型
# allowed_entity_types = ["PERSON", "DYNASTY", "EVENT", "PLACE"]
# allowed_relation_types = ["FOUNDED", "RULED", "BORN_IN", "DIED_IN", "BATTLED"]

# # 使用自定义提示模板和类型创建 KnowledgeGraphIndex
# kg_index = KnowledgeGraphIndex(
#     kg_triplet_extract_template=CUSTOM_NOUN_TRIPLET_EXTRACT_PROMPT,
#     allowed_entity_types=allowed_entity_types,
#     allowed_relation_types=allowed_relation_types
# )

