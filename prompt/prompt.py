mingchaonaxieshi_v1 = {
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

mingchaonaxieshi_v2 = {
    "CUSTOM_KG_TRIPLET_EXTRACT_TMPL": (
        "从给定文本中提取重大事件、关键人物和重要组织的信息，并着重提取时间信息，严格过滤无关内容。输出知识图谱前进行后处理，移除无效节点和关系。\n"
        "---------------------\n"
        "规则：\n"
        "- **仅保留以下内容：**\n"
        "  1. 重大事件：具有明确历史意义的事件（如靖难之役、土木堡之变）。\n"
        "  2. 关键人物：与事件或组织直接相关的重要人物（如朱重八、郭子兴）。\n"
        "  3. 重要组织：与事件或人物相关的历史组织（如红巾军、明朝）。\n"
        "- **时间信息（必选）**：\n"
        "  1. 优先提取具体时间信息（如“1348年”、“洪武元年”）。\n"
        "  2. 若时间信息模糊，则标注为“大约在某个时间”或“时间未知”。\n"
        "- **严格过滤**：\n"
        "  - 剔除情绪性描述（如“勇气”）、抽象描述（如“心理”）。\n"
        "  - 剔除背景性信息（如“生活困难”、“社会动荡”）。\n"
        "  - 剔除没有时间信息的三元组。\n"
        "\n"
        "步骤1：提取初步三元组\n"
        "- 提取与重大事件、关键人物、重要组织直接相关的三元组。\n"
        "- 输出格式：[{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': '', 'time': ''}}]\n"
        "\n"
        "步骤2：后处理\n"
        "- 检查三元组，剔除以下无效内容：\n"
        "  1. 无时间信息的三元组。\n"
        "  2. 无关节点：如情绪性描述（如“勇气”）、抽象描述（如“心理”）。\n"
        "  3. 无效关系：如仅描述状态、情绪或背景的复杂修饰。\n"
        "  4. 重复三元组：如多个相同的三元组。\n"
        "- 保留以下类型的三元组：\n"
        "  - 至少包含一个核心实体（事件、人物或组织）。\n"
        "  - 关系清晰，语义明确，具有直接相关性。\n"
        "  - 必须包含时间信息。\n"
        "\n"
        "示例：\n"
        "文本：朱重八于1348年加入了郭子兴领导的红巾军，并最终在1368年建立了明朝。\n"
        "初步生成：\n"
        "[{{'head': '朱重八', 'head_type': 'PERSON', 'relation': 'PARTICIPANT_IN', 'tail': '红巾军', 'tail_type': 'ORGANIZATION', 'time': '1348'}},\n"
        " {{'head': '郭子兴', 'head_type': 'PERSON', 'relation': 'LEADER_OF', 'tail': '红巾军', 'tail_type': 'ORGANIZATION', 'time': '1348'}},\n"
        " {{'head': '朱重八', 'head_type': 'PERSON', 'relation': 'FOUNDER_OF', 'tail': '明朝', 'tail_type': 'DYNASTY', 'time': '1368'}}]\n"
        "后处理结果：\n"
        "[{{'head': '朱重八', 'head_type': 'PERSON', 'relation': 'PARTICIPANT_IN', 'tail': '红巾军', 'tail_type': 'ORGANIZATION', 'time': '1348'}},\n"
        " {{'head': '郭子兴', 'head_type': 'PERSON', 'relation': 'LEADER_OF', 'tail': '红巾军', 'tail_type': 'ORGANIZATION', 'time': '1348'}},\n"
        " {{'head': '朱重八', 'head_type': 'PERSON', 'relation': 'FOUNDER_OF', 'tail': '明朝', 'tail_type': 'DYNASTY', 'time': '1368'}}]\n"
        "---------------------\n"
        "文本：{text}\n"
        "输出：\n"
    ),
    "allowed_entity_types": [
        "PERSON", "EVENT", "ORGANIZATION"
    ],
    "allowed_relation_types": [
        "PARTICIPANT_IN", "FOUNDER_OF", "LEADER_OF", "RESULTED_IN", "ALLY_OF", "OPPOSED_BY", "INVOLVED_IN"
    ]
}

mingchaonaxieshi_v3 = {
    "CUSTOM_KG_TRIPLET_EXTRACT_TMPL":(
        "从给定文本中提取与明朝历史相关的知识三元组，并补充时间信息。\n"
        "每个三元组应以 (head, relation, tail) 及其各自的类型形式出现，"
        "并尽可能添加明确的时间信息；如无明确时间，则进行时间推理。\n"
        "---------------------\n"
        "初始本体论：\n"
        "实体类型：{allowed_entity_types}\n"
        "关系类型：{allowed_relation_types}\n"
        "\n"
        "以这些类型为起点，但根据上下文需要引入新类型。\n"
        "\n"
        "指导原则：\n"
        "1. 输出格式严格为 JSON：[{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': '', 'time': ''}}]\n"
        "2. 时间提取：\n"
        "   - 尽可能从文本中提取明确的时间信息（年/月/日）。\n"
        "   - 如果文本中未明确提及时间，进行时间推理。例如：\n"
        "     - '1720年朱元璋开始流浪，3年后他成为皇帝' => '1723年朱元璋成为皇帝'\n"
        "     - '正德年间张居正改革' => 推断具体时间范围（正德年间：1505-1521）。\n"
        "   - 如果时间推理仍然不明确，标注为“未知时间”。\n"
        "3. 时间格式：时间字段应为 YYYY-MM-DD 或 YYYY-MM（若缺少日）或 YYYY（若缺少月和日）。\n"
        "4. 使用最完整的实体名称（例如'朱元璋'而不是'元璋'）。\n"
        "5. 保持实体简洁（最多 3-5 个词）。\n"
        "6. 对复杂事件分解为多个三元组。例如：\n"
        "   - 靖难之役导致了明成祖的登基和明初中央集权的加强 =>\n"
        "     [\n"
        "       {{'head': '靖难之役', 'relation': 'RESULTED_IN', 'tail': '明成祖登基', 'tail_type': 'EVENT', 'time': '1402'}},\n"
        "       {{'head': '靖难之役', 'relation': 'RESULTED_IN', 'tail': '中央集权加强', 'tail_type': 'SOCIAL_SYSTEM', 'time': '1402'}}\n"
        "     ]\n"
        "7. 特别关注以下内容：\n"
        "   - 重要历史事件（如'靖难之役'、'土木堡之变'）。\n"
        "   - 政策的背景、执行及其影响（如'张居正改革'）。\n"
        "   - 组织（如'东厂'）和机构（如'内阁'）。\n"
        "   - 明朝的重要人物（如'朱元璋'、'王阳明'）。\n"
        "8. 提取事件间的因果关系（如'土木堡之变导致明英宗被俘虏'）。\n"
        "9. 时间信息若可选推理多个结果，请选择最合理的时间范围，并标注推理来源。\n"
        "---------------------\n"
        "文本：{text}\n"
        "输出：\n"
    ),
    "allowed_entity_types":[
        "PERSON", "EVENT", "POLITICAL_FIGURE", "MILITARY_LEADER",
        "DYNASTY", "BATTLE", "CAPITAL", "POLICY", "INSTITUTION",
        "CULTURAL_FIGURE", "ECONOMIC_POLICY", "SOCIAL_SYSTEM",
        "DATE", "ORGANIZATION"
    ],
    "allowed_relation_types":[
        "FOUNDER", "MEMBER", "GENERAL", "CAPTURED_BY", "SUCCEEDED_BY",
        "PARTICIPANT", "RESULTED_IN", "IMPLEMENTED", "INFLUENCED_BY",
        "BORN_IN", "DIED_IN", "ASCENDED_THRONE", "STARTED_REIGN",
        "ENDED_REIGN", "LED_BATTLE", "DEFEATED", "ESTABLISHED",
        "DISSOLVED", "EXECUTED", "PROPOSED_POLICY", "IMPACTED"
    ]
}

mingchaonaxieshi_v4 = {
    "CUSTOM_KG_TRIPLET_EXTRACT_TMPL":(
        "从给定文本中提取与明朝历史相关的知识三元组，并补充时间信息。\n"
        "每个三元组应以 (head, relation, tail) 及其各自的类型形式出现，"
        "并尽可能添加明确的时间信息；如无明确时间，则进行时间推理。\n"
        "---------------------\n"
        "初始本体论：\n"
        "实体类型：{allowed_entity_types}\n"
        "关系类型：{allowed_relation_types}\n"
        "\n"
        "指导原则：\n"
        "1. 输出格式严格为 JSON：[{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': '', 'time': ''}}]\n"
        "2. 时间提取：\n"
        "   - 时间字段是三元组中的核心部分。\n"
        "   - 提取时间信息时，请遵循以下顺序：\n"
        "     a. 从原文中提取明确的时间表达（如年份、具体日期）。\n"
        "     b. 进行时间推理，例如：\n"
        "        - '1720年朱元璋开始流浪，3年后他成为皇帝' => '1723年朱元璋成为皇帝'\n"
        "        - '正德年间张居正改革' => 推断时间范围为 '1505-1521'。\n"
        "        - 如果涉及模糊时间（如“明初”），根据背景知识推断具体时间范围。\n"
        "     c. 如果时间推理仍不明确，则标注为“未知时间”。\n"
        "   - 时间字段应优先使用 ISO 格式：YYYY-MM-DD 或 YYYY-MM（若缺少日）或 YYYY（若缺少月和日）。\n"
        "   - 所有涉及动态事件的时间都必须标注。\n"
        "---------------------\n"
        "文本：{text}\n"
        "输出：\n"
    ),
    "allowed_entity_types":[
        "PERSON", "EVENT", "POLITICAL_FIGURE", "MILITARY_LEADER",
        "DYNASTY", "BATTLE", "CAPITAL", "POLICY", "INSTITUTION",
        "CULTURAL_FIGURE", "ECONOMIC_POLICY", "SOCIAL_SYSTEM",
        "DATE", "PERIOD", "TIMELINE", "ORGANIZATION"
    ],
    "allowed_relation_types":[
        "FOUNDER", "MEMBER", "GENERAL", "CAPTURED_BY", "SUCCEEDED_BY",
        "PARTICIPANT", "RESULTED_IN", "IMPLEMENTED", "INFLUENCED_BY",
        "BORN_IN", "DIED_IN", "ASCENDED_THRONE", "STARTED_REIGN",
        "ENDED_REIGN", "LED_BATTLE", "DEFEATED", "ESTABLISHED",
        "DISSOLVED", "EXECUTED", "PROPOSED_POLICY", "IMPACTED",
        "OCCURRED_IN", "DURATION", "HAPPENED_BEFORE", "HAPPENED_AFTER"
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

# 物理教材模版
highschool_physics = {
    "CUSTOM_KG_TRIPLET_EXTRACT_TMPL": (
                "从给定的文本中提取多层次的物理学知识三元组，"
                "包括物理量、公式、定理、物理过程、物理现象、题目、答案等，"
                "每个三元组以 (head, relation, tail) 及其各自的类型形式出现。\n"
                "---------------------\n"
                "初始本体论：\n"
                "实体类型：{allowed_entity_types}\n"
                "关系类型：{allowed_relation_types}\n"
                "指导原则：\n"
                "- 以 JSON 格式输出：[{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
                "- 使用最完整的实体形式（例如 '牛顿第二定律' 而不是 '牛顿定律'）\n"
                "- 对复杂的物理过程进行分解，例如：\n"
                "  1. '力的作用使物体发生加速度' =>\n"
                "     [{{'head': '力', 'relation': 'CAUSES', 'tail': '加速度', 'tail_type': 'PHYSICAL_QUANTITY'}}]\n"
                "  2. '重力作用于物体使其下落' =>\n"
                "     [{{'head': '重力', 'relation': 'ACTS_ON', 'tail': '物体', 'tail_type': 'OBJECT'}},\n"
                "      {{'head': '物体', 'relation': 'MOVES_IN', 'tail': '下落', 'tail_type': 'PHENOMENON'}}]\n"
                "- 提取物理公式及其相关解释，确保公式的准确性和上下文的连贯性。\n"
                "- 捕捉教材中的例题和解题过程，以便建立物理现象与公式之间的联系。\n"
                "- 提取教材的背景信息（例如章节、作者、主题），为知识图谱提供上下文。\n"
                "---------------------\n"
                "文本：{text}\n"
                "输出：\n"
            ),
    "allowed_entity_types": [
        "PHYSICAL_QUANTITY", "FORMULA", "LAW", "THEORY", "CONCEPT", "OBJECT",
        "PROCESS", "EXAMPLE", "QUESTION", "ANSWER", "EXPERIMENT", "PHENOMENON", "CONDITION"
    ],
    "allowed_relation_types": [
        "CAUSES", "ACTS_ON", "RELATED_TO", "APPLIES_TO", "DERIVED_FROM", "INVOLVES", "EQUALS",
        "MEASURED_IN", "PART_OF", "RESULTS_IN", "USEFUL_IN", "IMPACTED_BY", "PROVIDES_EXAMPLE",
        "IS_QUESTION_FOR", "HAS_ANSWER", "EXPLAINS", "DEPENDS_ON", "HAS_UNIT"
    ]
}

novel_kg_extraction = {
    "CUSTOM_KG_TRIPLET_EXTRACT_TMPL": (
                "从给定的小说文本中提取多层次的知识三元组，"
                "包括人物关系、事件描述、地点描述、物品描述、时间描述、情节发展等，"
                "每个三元组以 (head, relation, tail) 及其各自的类型形式出现。\n"
                "---------------------\n"
                "初始本体论：\n"
                "实体类型：{allowed_entity_types}\n"
                "关系类型：{allowed_relation_types}\n"
                "指导原则：\n"
                "- 以 JSON 格式输出：[{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
                "- 使用最完整的实体形式（例如 '伊丽莎白·班内特' 而不是 '伊丽莎白'）。\n"
                "- 对复杂的事件或情节进行分解，例如：\n"
                "  1. '达西向伊丽莎白求婚并被拒绝' =>\n"
                "     [{{'head': '达西', 'relation': 'PROPOSED_TO', 'tail': '伊丽莎白·班内特', 'tail_type': 'CHARACTER'}},\n"
                "      {{'head': '伊丽莎白·班内特', 'relation': 'REJECTED', 'tail': '达西', 'tail_type': 'CHARACTER'}}]\n"
                "  2. '一场暴风雨袭击了村庄' =>\n"
                "     [{{'head': '暴风雨', 'relation': 'IMPACTED', 'tail': '村庄', 'tail_type': 'LOCATION'}}]\n"
                "  3. '战争在1800年爆发' =>\n"
                "     [{{'head': '战争', 'relation': 'HAPPENED_IN', 'tail': '1800年', 'tail_type': 'TIME'}}]\n"
                "- 提取人物之间的关系，包括家庭关系、友谊、敌对关系等。\n"
                "- 提取小说中的重要地点及其相关事件。\n"
                "- 提取重要物品及其在情节中的作用，例如宝物、信件等。\n"
                "- 提取章节或段落中的背景信息（例如章节名、作者、主题），为知识图谱提供上下文。\n"
                "- 避免冗余三元组，确保输出清晰、简洁。\n"
                "---------------------\n"
                "文本：{text}\n"
                "输出：\n"
            ),
    "allowed_entity_types": [
        "CHARACTER", "EVENT", "LOCATION", "OBJECT", "EMOTION", "RELATIONSHIP", "CHAPTER", "CONCEPT", "TIME", "ORGANIZATION", "SYMBOL", "MOTIF", "ACTION"
    ],
    "allowed_relation_types": [
        "PROPOSED_TO", "REJECTED", "FRIENDS_WITH", "ENEMY_OF", "LOCATED_IN", "HAPPENED_IN", 
        "POSSESSES", "DELIVERED_TO", "INVOLVES", "DEPENDS_ON", "CAUSED", "IMPACTED", 
        "MENTIONED_IN", "PART_OF", "HAS_EMOTION", "EXPLAINS", "OCCURRED_AT", "REPRESENTS", 
        "ASSOCIATED_WITH", "BELONGS_TO", "INFLUENCES", "INSPIRED_BY"
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

