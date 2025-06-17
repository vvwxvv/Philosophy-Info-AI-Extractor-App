"""Enhanced philosophical schools definitions"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from philosophy_knowledge_base import PhilosophicalEntity


@dataclass
class PhilosophicalSchool(PhilosophicalEntity):
    """Represents a philosophical school or tradition"""

    founders: List[Dict[str, str]] = field(default_factory=list)
    key_figures: List[Dict[str, str]] = field(default_factory=list)
    core_tenets: List[Dict[str, str]] = field(default_factory=list)
    historical_period: Dict[str, str] = field(default_factory=dict)
    geographical_origin: Dict[str, str] = field(default_factory=dict)
    influences: List[Dict[str, str]] = field(default_factory=list)
    influenced: List[Dict[str, str]] = field(default_factory=list)
    key_texts: List[Dict[str, str]] = field(default_factory=list)
    practices: List[Dict[str, str]] = field(default_factory=list)
    contemporary_relevance: Dict[str, str] = field(default_factory=dict)
    sub_schools: List[str] = field(default_factory=list)
    opposing_schools: List[str] = field(default_factory=list)


def initialize_schools() -> Dict[str, PhilosophicalSchool]:
    """Initialize philosophical schools"""
    schools = {}

    # Stoicism
    schools["stoicism"] = PhilosophicalSchool(
        id="stoicism",
        name={
            "en": "Stoicism",
            "zh": "斯多葛主义",
            "de": "Stoa",
            "fr": "Stoïcisme",
            "gr": "Στωικισμός",
        },
        description={
            "en": "A Hellenistic philosophy emphasizing virtue, reason, and living in accordance with nature",
            "zh": "一个强调美德、理性和按照自然生活的希腊化哲学流派",
        },
        founders=[
            {
                "en": "Zeno of Citium (334-262 BCE) - Founded the school in Athens",
                "zh": "季蒂昂的芝诺 (公元前334-262) - 在雅典创立了这个学派",
            }
        ],
        key_figures=[
            {
                "en": "Epictetus (50-135 CE) - Developed Stoic ethics and practice",
                "zh": "爱比克泰德 (公元50-135) - 发展了斯多葛伦理学和实践",
            },
            {
                "en": "Marcus Aurelius (121-180 CE) - Roman Emperor and Stoic philosopher",
                "zh": "马可·奥勒留 (公元121-180) - 罗马皇帝和斯多葛哲学家",
            },
            {
                "en": "Seneca (4 BCE-65 CE) - Roman Stoic and advisor to Nero",
                "zh": "塞涅卡 (公元前4-公元65) - 罗马斯多葛派和尼禄的顾问",
            },
            {
                "en": "Chrysippus (279-206 BCE) - Systematized Stoic logic",
                "zh": "克律西波斯 (公元前279-206) - 系统化了斯多葛逻辑",
            },
        ],
        core_tenets=[
            {
                "en": "Virtue is the sole good - Living virtuously is the path to eudaimonia",
                "zh": "美德是唯一的善 - 有德行的生活是通向幸福的道路",
            },
            {
                "en": "Dichotomy of control - Focus only on what is within your control",
                "zh": "控制二分法 - 只关注你能控制的事物",
            },
            {
                "en": "Cosmic sympathy - All things are interconnected in the universe",
                "zh": "宇宙共鸣 - 万物在宇宙中相互联系",
            },
            {
                "en": "Apatheia - Freedom from destructive emotions through reason",
                "zh": "无欲 - 通过理性摆脱破坏性情绪",
            },
            {
                "en": "Amor fati - Love and accept one's fate",
                "zh": "爱命运 - 热爱并接受自己的命运",
            },
        ],
        historical_period={
            "en": "3rd century BCE to 3rd century CE (Ancient Period), Revival in Renaissance and Modern times",
            "zh": "公元前3世纪至公元3世纪（古代时期），在文艺复兴和现代复兴",
        },
        geographical_origin={
            "en": "Athens, Greece, later spread throughout the Roman Empire",
            "zh": "希腊雅典，后来传播到整个罗马帝国",
        },
        key_texts=[
            {"en": "Meditations by Marcus Aurelius", "zh": "马可·奥勒留的《沉思录》"},
            {
                "en": "Discourses and Enchiridion by Epictetus",
                "zh": "爱比克泰德的《论说集》和《手册》",
            },
            {
                "en": "Letters and Essays by Seneca",
                "zh": "塞涅卡的《书信集》和《论文集》",
            },
        ],
        practices=[
            {
                "en": "Morning reflection - Planning the day with virtue in mind",
                "zh": "晨思 - 以美德为念规划一天",
            },
            {
                "en": "Evening review - Reflecting on the day's actions and thoughts",
                "zh": "晚省 - 反思当天的行为和思想",
            },
            {
                "en": "Negative visualization - Contemplating loss to appreciate what you have",
                "zh": "消极想象 - 思考失去以珍惜拥有",
            },
            {
                "en": "View from above - Seeing things from a cosmic perspective",
                "zh": "俯瞰视角 - 从宇宙角度看待事物",
            },
        ],
        contemporary_relevance={
            "en": "Influenced modern CBT, resilience training, and self-help philosophy",
            "zh": "影响了现代认知行为疗法、韧性训练和自助哲学",
        },
        sub_schools=["early_stoa", "middle_stoa", "late_stoa", "neo_stoicism"],
        opposing_schools=["epicureanism", "skepticism", "cynicism"],
        keywords=["virtue", "reason", "nature", "control", "wisdom", "resilience"],
        aliases=["Stoa", "Stoic philosophy", "Stoic school"],
    )

    # Existentialism
    schools["existentialism"] = PhilosophicalSchool(
        id="existentialism",
        name={
            "en": "Existentialism",
            "zh": "存在主义",
            "de": "Existentialismus",
            "fr": "Existentialisme",
        },
        description={
            "en": "A philosophical movement emphasizing individual existence, freedom, and the search for meaning",
            "zh": "一个强调个体存在、自由和寻找意义的哲学运动",
        },
        founders=[
            {
                "en": "Søren Kierkegaard (1813-1855) - Often considered the father of existentialism",
                "zh": "索伦·克尔凯郭尔 (1813-1855) - 通常被认为是存在主义之父",
            }
        ],
        key_figures=[
            {
                "en": "Jean-Paul Sartre (1905-1980) - Developed atheistic existentialism",
                "zh": "让-保罗·萨特 (1905-1980) - 发展了无神论存在主义",
            },
            {
                "en": "Martin Heidegger (1889-1976) - Analyzed the nature of Being",
                "zh": "马丁·海德格尔 (1889-1976) - 分析了存在的本质",
            },
            {
                "en": "Simone de Beauvoir (1908-1986) - Applied existentialism to feminism",
                "zh": "西蒙娜·德·波伏娃 (1908-1986) - 将存在主义应用于女性主义",
            },
            {
                "en": "Albert Camus (1913-1960) - Explored absurdism and revolt",
                "zh": "阿尔贝·加缪 (1913-1960) - 探索了荒诞主义和反抗",
            },
        ],
        core_tenets=[
            {
                "en": "Existence precedes essence - We create our own essence through choices",
                "zh": "存在先于本质 - 我们通过选择创造自己的本质",
            },
            {
                "en": "Radical freedom - Humans are condemned to be free",
                "zh": "彻底的自由 - 人类注定是自由的",
            },
            {
                "en": "Authenticity - Living in accordance with one's true self",
                "zh": "真实性 - 按照真实的自我生活",
            },
            {
                "en": "Anxiety and dread - Confronting the absurdity of existence",
                "zh": "焦虑和恐惧 - 面对存在的荒诞性",
            },
            {
                "en": "Bad faith - Self-deception to avoid responsibility",
                "zh": "恶意 - 自欺以逃避责任",
            },
        ],
        historical_period={
            "en": "19th-20th century, peaked after World War II",
            "zh": "19-20世纪，在第二次世界大战后达到高峰",
        },
        key_texts=[
            {"en": "Being and Time by Heidegger", "zh": "海德格尔的《存在与时间》"},
            {"en": "Being and Nothingness by Sartre", "zh": "萨特的《存在与虚无》"},
            {"en": "The Second Sex by de Beauvoir", "zh": "波伏娃的《第二性》"},
            {"en": "The Myth of Sisyphus by Camus", "zh": "加缪的《西西弗斯神话》"},
        ],
        contemporary_relevance={
            "en": "Influences psychology, literature, theology, and popular culture's focus on authenticity",
            "zh": "影响心理学、文学、神学和流行文化对真实性的关注",
        },
        sub_schools=[
            "atheistic_existentialism",
            "christian_existentialism",
            "absurdism",
        ],
        keywords=[
            "existence",
            "freedom",
            "authenticity",
            "anxiety",
            "choice",
            "meaning",
        ],
    )

    # Add more schools...

    return schools
