"""Philosophy categories and types definitions"""

PHILOSOPHY_CATEGORIES = {
    "形而上学 / Metaphysics": [
        "存在 / being",
        "本体论 / ontology",
        "实在 / reality",
        "本质 / essence",
        "因果 / causality",
        "时间 / time",
        "空间 / space",
        "意识 / consciousness"
    ],
    "认识论 / Epistemology": [
        "知识 / knowledge",
        "真理 / truth",
        "信念 / belief",
        "理性 / reason",
        "经验 / experience",
        "感知 / perception",
        "怀疑 / skepticism",
        "确定性 / certainty"
    ],
    "伦理学 / Ethics": [
        "道德 / morality",
        "价值 / value",
        "责任 / responsibility",
        "正义 / justice",
        "自由 / freedom",
        "美德 / virtue",
        "幸福 / happiness",
        "义务 / duty"
    ],
    "政治哲学 / Political Philosophy": [
        "权力 / power",
        "民主 / democracy",
        "权利 / rights",
        "平等 / equality",
        "自由 / liberty",
        "正义 / justice",
        "国家 / state",
        "社会契约 / social contract"
    ],
    "美学 / Aesthetics": [
        "美 / beauty",
        "艺术 / art",
        "品味 / taste",
        "崇高 / sublime",
        "创造力 / creativity",
        "表现 / expression",
        "形式 / form",
        "体验 / experience"
    ],
    "逻辑学 / Logic": [
        "推理 / reasoning",
        "论证 / argument",
        "有效性 / validity",
        "真值 / truth value",
        "谬误 / fallacy",
        "演绎 / deduction",
        "归纳 / induction",
        "形式逻辑 / formal logic"
    ],
    "语言哲学 / Philosophy of Language": [
        "意义 / meaning",
        "指称 / reference",
        "真理 / truth",
        "理解 / understanding",
        "交流 / communication",
        "符号 / symbol",
        "解释 / interpretation",
        "语境 / context"
    ],
    "心灵哲学 / Philosophy of Mind": [
        "意识 / consciousness",
        "心灵 / mind",
        "身体 / body",
        "认知 / cognition",
        "意向性 / intentionality",
        "感受质 / qualia",
        "自我 / self",
        "人工智能 / artificial intelligence"
    ]
}

PHILOSOPHY_EVENT_TYPES = [
    r'(讲座|lecture)',
    r'(研讨会|seminar)',
    r'(会议|conference)',
    r'(工作坊|workshop)',
    r'(辩论|debate)',
    r'(读书会|reading group)',
    r'(对话|dialogue)',
    r'(公开课|public course)'
]

PHILOSOPHY_FIELDS_TYPE = [
    "Look for text within quotation marks 《》 or quotes",
    "Text following '概念:', '理论:', '论证:', '学派:'",
    "Titles near philosophical terms or concepts"
] 