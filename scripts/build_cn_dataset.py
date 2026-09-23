"""构建中文微调/评估数据集（与 LocalLLaMA/typed-decisions 相同的 state/questions/gold 格式）。

三个任务与官方工作流对齐：
  intent   (choice, 5 选项)  客服工单意图分类
  urgency  (score, 4 级)     工单紧急度
  churn    (noul)            是否有流失风险

生成方式：模板 + 槽位填充，标签由生成逻辑决定（确定性）。
  训练集  data/cn_train.jsonl  （模板生成，1200 条）
  测试集  data/cn_test.jsonl   （模板生成 260 条，不同槽位/种子）
  困难集  data/cn_test_hard.jsonl（手写自然句，80 条，仅评估用）
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

INTENT_CRITERIA = {
    "refund": "申请退款、退货或换货",
    "logistics": "查询物流进度、修改收货地址或配送问题",
    "product": "产品功能咨询、使用方法或质量故障",
    "complaint": "对服务态度或处理结果不满的投诉",
    "other": "其他一般性咨询",
}
URGENCY_CRITERIA = [
    "一般咨询，可隔日处理",
    "有一定影响，建议当日处理",
    "影响正常使用，需尽快处理",
    "严重故障或资金损失，需立即处理",
]

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "判断这位用户的主要诉求属于哪一类。",
        "criteria": INTENT_CRITERIA,
    },
    "urgency": {
        "type": "score",
        "instructions": "评估该工单的处理紧急程度（0-3）。",
        "criteria": URGENCY_CRITERIA,
    },
    "churn": {
        "type": "noul",
        "instructions": "这位用户是否存在流失（退订/转投竞争对手）的风险？",
        "criteria": {"true": "存在流失风险", "false": "不存在明显流失风险"},
    },
}

NAMES = ["王女士", "李先生", "张同学", "刘女士", "陈先生", "赵女士", "孙先生", "周女士", "吴先生", "郑女士"]
PRODUCTS = ["无线耳机", "智能手表", "笔记本电脑", "空气净化器", "扫地机器人", "电饭煲", "电动牙刷", "移动硬盘"]
CITIES = ["杭州", "成都", "武汉", "南京", "西安", "重庆", "苏州", "长沙", "青岛", "宁波"]
ISSUES_PRODUCT = ["充不上电", "连接蓝牙总是断开", "开机后黑屏", "噪音很大", "按键失灵", "频繁死机重启"]
ISSUES_LOGISTICS = ["显示已发货但一周没有物流更新", "物流显示签收但我没有收到货", "快递在中转站滞留了三天", "配送日期一拖再拖"]

SEV_HIGH = ["完全无法使用", "一点反应都没有", "数据全部丢失", "把我重要文件弄丢了"]
SEV_MID = ["严重影响使用", "耽误了我的工作", "马上要开会要用"]
CHURN_TRUE = [
    "再这样我就卸载了",
    "处理不好我就转投别家了",
    "我用你们平台三年了，真是越来越失望，再不行就只能换一家了",
    "我已经在对比其他品牌了",
    "朋友推荐了别家的，我本来还想再给你们一次机会",
]
CHURN_FALSE_SOFTEN = ["你们家产品我一直用着挺习惯的", "我是老用户了，相信你们能处理好", "麻烦了，谢谢"]

TEMPLATES = {
    "refund": [
        "{name}上个月在你们平台买了一台{product}，用了没几天就{issue}，{sevopt}我要退款，订单号{order}。",
        "你好，我买的{product}和商品页描述不符，我申请退货退款，订单号是{order}。",
        "{order}这个订单的{product}我-seven-天内无理由，请帮我办理退款。{churnopt}",
    ],
    "logistics": [
        "我的订单{order}已经{logissue}，收货地址是{city}，帮我查一下怎么回事。",
        "{name}想修改收货地址，订单{order}原本寄到{city}，现在想改寄{city2}，快递明天就发了，麻烦尽快处理。",
        "你们快递到底怎么回事？订单{order}{logissue}。{churnopt}",
    ],
    "product": [
        "你们的{product}{issue}，说明书上也没写怎么解决，这个到底怎么用？",
        "想咨询一下{product}的保修政策，另外它{issue}是不是质量问题？订单号{order}。",
        "{product}刚过保就{issue}，这算不算设计缺陷？{churnopt}",
    ],
    "complaint": [
        "你们客服{agent}态度太差了，我的问题反映了三次都没解决，我要投诉！{churnopt}",
        "投诉！昨天打电话承诺当天回复，到现在也没人理我，订单{order}的问题一直拖着。",
        "对你们的服务非常不满意，{agent}承诺的补偿一直不兑现，请给我一个说法。",
    ],
    "other": [
        "请问你们周末营业吗？想了解一下会员有什么优惠活动。",
        "你们家的{product}和上一代比有什么升级？{softenopt}",
        "想问下积分怎么兑换优惠券，谢谢。",
    ],
}


def _fill(template, rng):
    sevopt = rng.choice(SEV_HIGH + SEV_MID + [""])
    churnopt = rng.choice(CHURN_TRUE + [""])
    softenopt = rng.choice(CHURN_FALSE_SOFTEN + [""])
    return (template
            .replace("{name}", rng.choice(NAMES))
            .replace("{product}", rng.choice(PRODUCTS))
            .replace("{city2}", rng.choice(CITIES))
            .replace("{city}", rng.choice(CITIES))
            .replace("{order}", "GD%08d" % rng.randint(0, 99999999))
            .replace("{agent}", rng.choice(["小A", "工号8867的客服", "在线客服"]))
            .replace("{issue}", rng.choice(ISSUES_PRODUCT))
            .replace("{logissue}", rng.choice(ISSUES_LOGISTICS))
            .replace("{sevopt}", sevopt + ("，" if sevopt else ""))
            .replace("{churnopt}", (("，" if rng.random() < 0.5 else "") + churnopt) if churnopt else "")
            .replace("{softenopt}", ((", " + softenopt) if softenopt else ""))
            .replace("-seven-", "7"))


def gold_for(intent, urgency, churn):
    """返回与 typed-decisions gold 结构一致的软目标分布（近似 one-hot）。"""
    def onehot(k, n):
        p = [0.03 / (n - 1)] * n
        p[k] = 0.97
        return p
    return {
        "intent": {"probabilities": dict(zip(INTENT_CRITERIA.keys(), onehot(list(INTENT_CRITERIA).index(intent), 5)))},
        "urgency": {"probabilities": {str(i): v for i, v in enumerate(onehot(urgency, 4))}},
        "churn": {"probabilities": {"false": 1 - churn if isinstance(churn, float) else (0.03 if churn else 0.97),
                                    "true": churn if isinstance(churn, float) else (0.97 if churn else 0.03)}},
    }


def urgency_of(intent, state, rng):
    if intent == "other":
        return 0
    if any(k in state for k in SEV_HIGH):
        return rng.choice([2, 3])
    if any(k in state for k in SEV_MID):
        return 2
    if intent in ("refund", "logistics"):
        return rng.choice([1, 2])
    if intent == "complaint":
        return rng.choice([1, 2, 2])
    return 1


def churn_of(intent, state, rng):
    has_threat = any(k in state for k in CHURN_TRUE)
    if has_threat:
        return True
    if intent == "complaint" and rng.random() < 0.25:
        return True
    return False


def make_example(rng):
    for _ in range(20):
        intent = rng.choice(list(TEMPLATES))
        state = _fill(rng.choice(TEMPLATES[intent]), rng).strip("，。 ").strip()
        if len(state) < 12:
            continue
        urgency = urgency_of(intent, state, rng)
        churn = churn_of(intent, state, rng)
        return {
            "state": state,
            "questions": QUESTIONS,
            "gold": gold_for(intent, urgency, churn),
            "meta": {"intent": intent, "urgency": urgency, "churn": churn, "kind": "template"},
        }
    raise RuntimeError("failed to build an example")


# ---------------- 手写困难集（检验泛化，不参与训练） ----------------
# (state, intent, urgency, churn)
HARD = [
    ("昨天下午下的单，今天早上就到了，但是包装破了一个角，里面屏幕有裂痕，这种情况下换货运费谁出？", "refund", 3, False),
    ("你好，我上个月买的那台机器最近总是自动关机，重启之后过一会儿又关，是不是电池的问题，可以寄回检测吗？", "product", 2, False),
    ("三笔订单合并发的货，现在其中一件要退，运费怎么算？另外两件照常收货。", "refund", 1, False),
    ("物流信息五天没动了，客服电话排队四十分钟被挂断，在线客服只会发机器人话术，我已经忍无可忍。", "complaint", 3, True),
    ("帮我把明天的配送改到后天下午，家里没人。地址不变，还是小区丰巢柜那栋。", "logistics", 0, False),
    ("请问这次的以旧换新补贴和店铺优惠券可以叠加使用吗？", "other", 0, False),
    ("你们这个APP更新之后连不上设备了，重装了两次也没用，我工作全靠它，今天必须给我解决！", "product", 3, True),
    ("上次投诉你们说三个工作日回复，现在第七天了，一条消息都没有，就这服务态度还谈什么口碑？", "complaint", 2, True),
    ("耳机左边不出声了，刚过保半个月，能不能通融一下保修？我是五年老用户，家里设备全是你们家的。", "product", 1, False),
    ("快递员把我件放代收点都没发短信，我自己去找了半天，这种服务真的无语，麻烦登记一下。", "logistics", 1, False),
    ("想问下发票抬头开错了，能重新开吗？上个月的单子。", "other", 0, False),
    ("账户里那笔退款说三个工作日到账，现在都十天了还没到，是不是你们挪用了？再不到账我只能报警加投诉了。", "refund", 3, True),
    ("新买的加湿器水位满了还一直报缺水警报，是不是传感器坏了？退货运费你们承担吗？", "refund", 2, False),
    ("我地址填错了写成了前公司的，货还没发，麻烦拦截改地址，扣了运费我认。", "logistics", 1, False),
    ("我就想问，宣传页上写的续航十二小时，实际用四个小时就没电，这算不算虚假宣传？我要求退货并且你们出运费。", "refund", 2, True),
    ("客服说补我五十块优惠券，一周了还没到账，麻烦帮我催一下，谢谢。", "complaint", 1, False),
    ("你们仓库是不是漏发货了？箱子里就一张说明书，少了一个主机，订单显示已全部签收。", "logistics", 3, False),
    ("使用教程的视频链接点开全是404，产品是新手上手有点难，能发我一份图文版吗？", "product", 0, False),
    ("本来今天生日想着用你们家的火锅券，结果显示门店停业，客服也找不到人，太让人失望了。", "complaint", 1, True),
    ("帮我看下订单到哪了，显示在转归班", "logistics", 0, False),
    ("手机壳颜色和图片差太多，塑料感很重，虽然便宜也不想留，七天无理由走一下。", "refund", 0, False),
    ("APP会员自动续费扣款了，我明明上个月就关闭了续费，这算乱扣费吧？马上退给我，不然我直接向消协投诉。", "refund", 3, True),
    ("家里老人不会用你们的大字模式，每次都要我远程教，有没有更简单的操作说明？", "product", 0, False),
    ("订单显示拒收，但我根本没有拒收，现在货在哪都查不到，你们给我一个说法，不然我就下单别家了。", "logistics", 3, True),
    ("评价里说赠品要收货后七天内联系客服领取，我联系了两次都说记录一下，到现在没消息。", "complaint", 0, False),
    ("想咨询企业采购有没有批量折扣，大概五十台，需要开增值税专用发票。", "other", 0, False),
    ("净水器出水越来越慢，换了滤芯也没改善，售后说要收上门费，机器才买八个月，这合理吗？", "product", 2, False),
    ("地址写错了市和区反了，已经发货了还能改吗？改不了我就拒收重新拍，运费险还在有效期内。", "logistics", 1, False),
    ("用了两年第一次遇到这种事：换货的新机器发来屏幕上有划痕，你们质检是不是形同虚设？这次必须给我个满意的处理，不然我只能发帖曝光+退货。", "refund", 3, True),
    ("问一下这个型号的充电器功率，我的旧充电头能不能通用？", "other", 0, False),
    ("退款进度一直显示财务审核中，客服说加急了，但已经加急两次了，我信用卡账单都要出了。", "refund", 2, False),
    ("师傅上门装完就走，说明书也没留，遥控器怎么配对我都不会，打电话售后一直占线。", "product", 1, False),
    ("你们派送的快递员态度特别横，敲门跟砸门一样，说了几次都这样，请换个人派送或者投诉他。", "complaint", 1, False),
    ("会员日的价买完第二天就降价了，能补个差价吗？不能补的话我就退了重买，反正七天无理由。", "refund", 1, True),
    ("商品页面写的现货，付款之后告诉我没货要等一个月，这不是欺骗吗？退！顺便把我的优惠券恢复一下，是你们让取消的。", "refund", 2, True),
    ("领优惠券的时候提示活动太火爆，试了半小时都领不到，是不是根本没放量？", "other", 0, False),
    ("转运的时候把我的件和别人的搞混了，我拿到的是一箱猫粮，我的机票订单（其实是行李箱）不见了，你们抓紧查！", "logistics", 2, False),
    ("儿童手表定位漂移严重，学校在科技园它显示在三公里外，老人根本看不懂，这涉及安全功能，请尽快优化或给我换货。", "product", 3, False),
    ("用了三年从来没出过问题，这次风扇声音突然很大，麻烦约个上门检测，周末我在家。", "product", 1, False),
    ("你们的以旧换新上门取件约了两次都放我鸽子，旧机器一直占着地方，再约不上我就当别家的回收了。", "logistics", 2, True),
    (" developing countries 帮我查下这个单号，平台外发的，显示清关异常一周了", "logistics", 1, False),
    ("客服给我答复问题时全程用英文，我说了很多次听不懂，还是继续发英文，这是哪门子服务？", "complaint", 1, False),
    ("行车记录仪内存卡老是提示要格式化，格式化完数据就没了，重要影像全丢，你们这个设计有没有考虑过用户？我要求换新并保留追究权利。", "complaint", 3, True),
    ("抱着试试的心态买的，没想到电池确实耐用，就是想夸夸，顺便问问延保怎么买。", "other", 0, False),
    ("发货太快了点赞，但是发错了颜色，白色拍成黑色发来，换货的运费和差价怎么处理？", "logistics", 1, False),
    ("退款已收到，速度挺快，不过之前客服说赔的十块钱红包还没发，能补一下吗？不着急。", "other", 0, False),
    ("血压计每次测的数值差别很大，跟医院设备对比差了十几个点，这精度达标吗？我要退货。", "refund", 2, False),
    ("快递直接放门口被别人拿走了，监控显示是个穿外卖服的人拿的，你们派件前连个电话都不打，责任谁负？", "logistics", 3, False),
    ("上次换货之后新的又有问题，同一批是不是都有缺陷？我不想再换了，直接退款吧，这品牌我是敬而远之了。", "refund", 2, True),
    ("问一下清洗滤网的周期是多久，说明书丢了，App里也找不到。", "product", 0, False),
    ("订单拆成两个包裹，第一个到了，第二个卡在广州五天，里面的东西下周 wedding 要用，麻烦催一下物流或者帮我催发。", "logistics", 2, False),
    ("你们的会员客服和普通客服有什么区别？充值年费值得吗？", "other", 0, False),
    ("投诉你们虚假发货！单号生成了三天物流公司说根本没收到件，就是为了超时考核吧？已经截图留证，再拖我就向平台监管投诉。", "complaint", 3, True),
    ("买的净化器滤芯和主机型号不匹配装不进去，页面没写清楚适配型号，这算谁的责任？我换货可以，但运费得你们出。", "refund", 1, False),
    ("老人手机紧急呼叫键按了没反应，这要是出事就是人命关天，你们今天必须给我答复怎么处理。", "product", 3, False),
    ("上次的问题最终是这样解决的，我来说下过程希望你们改进：第一次派单员不专业，第二次重派就好了，整体态度还行。", "other", 0, False),
    ("我买的两台设备绑定在同一个账号，其中一台想转让给朋友，怎么解绑？解绑后我的会员权益还在吗？", "other", 0, False),
    ("快递放驿站三天了不派送，我腿脚不便出门困难，打电话给网点说让我自己取，你们这就是歧视行动不便的用户？", "complaint", 2, True),
    ("游戏手柄连电脑识别不了，连手机正常，驱动也装了，是不是不兼容我的系统版本？", "product", 1, False),
    ("价格保护条款写的三十天内降价可补差，我申请你们说活动价不算，那条款写着给谁看的？要么补差要么退我按原价重拍的差价，我已经重新下单了。", "refund", 2, True),
    (" mirror 一下我的问题：预售的尾款付了，到现在不发货也不解释，客服就一句正在安排，你们就这样对待预售用户？", "complaint", 2, False),
    ("买了你们推荐搭配的扩展坞，插上就烧了接口，把我的电脑主板也烧了，维修报价三千多，你们必须负责到底，否则法庭见。", "complaint", 3, True),
    ("给爸妈买的老花镜度数和验光单不一致，戴着头晕，退货运费险只有十块，运费要十五，这五块你们能不能报销？", "refund", 1, False),
    ("每周三的直播课在哪看？我买了课但是找不到入口，客服说刷新页面就行，还是不行。", "product", 0, False),
    ("发出的货是翻新机！封条是后贴的，序列号查询激活时间在半年前，二手当新卖，你们胆子太大了，我已经联系12315，就等你们处理态度了。", "complaint", 3, True),
    (" suction 强度不错，就是想反馈下尘杯卡扣设计太紧，老人打开费劲，建议改进，另外滤网多少钱一个？", "other", 0, False),
    ("订单已发货但收件人手机号少了一位，联系不上快递员，麻烦帮忙改号或者拦截，别给我退回去了。", "logistics", 2, False),
    ("预付了定金现在不想要了，定金能退吗？知道可能退不了，就想确认下规则，别耽误我 seconds 内补尾款的判断。", "refund", 0, False),
    ("你们的售后工单系统是不是有bug？我提交的工单每次都自动关闭，显示已解决，可我的问题一次都没人联系过我，已经第四次提交了。", "complaint", 2, True),
    ("食物垃圾处理器装完之后下水比以前还慢，是不是你们推荐的型号跟我的老房子管道不匹配？申请退货检测。", "refund", 1, False),
    ("问下活力28那款洗衣液的保质期到什么时候，页面没写生产日期，家里有小孩比较在意这个。", "other", 0, False),
    ("用了你们的平衡车半年，这周突然断电，人从车上摔下来了，膝盖缝了三针。安全问题上你们必须书面回复原因和处理方案，不然我走法律途径并找媒体。", "complaint", 3, True),
    ("快递到了但是让我付九块钱运费，页面明明写着包邮，是不是商家把到付当包邮发？我先拒收了。", "logistics", 1, False),
    ("手环睡眠监测最近完全不记录了，重启恢复一天又不行，app里社区好多人反映新固件的问题，什么时候修复？", "product", 1, False),
    ("你们说换货免运费，结果上门取件师傅跟我要二十块上门费，到底哪个环节出了问题？钱不多但这事让我很反感。", "complaint", 1, False),
    ("之前买的台灯灯罩发黄，客服二话不说就补发了新的，给你们的服务点赞！顺路问问灯管配件能单买吗？", "other", 0, False),
    ("退款退到已注销的银行卡了怎么办？银行说会退回，你们这边能帮我改收款账户重新发起吗？急用这笔钱交房租。", "refund", 2, False),
    ("新家全屋设备绑定，发现你们智能家居跟我家已有的中控协议不兼容，问了一圈客服没人说得清支持哪些协议，这种基础信息为什么页面上不写明？", "product", 1, False),
    ("买了三次次次都有问题：第一次发错色，第二次有划痕，这次直接开不了机。我认了，退了之后不会再买你们任何东西，朋友问起我也会劝退。", "refund", 2, True),
    ("证件照打印机打印出来有色差，深蓝色印成紫色，调了色彩校正还是不行，这正常吗还是机器有问题？", "product", 1, False),
]

CHURN_OVERRIDES = {  # 手写集中个别标签覆盖自动判断
}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d rows -> %s" % (len(rows), path))


def main():
    os.makedirs(DATA, exist_ok=True)
    rng = random.Random(20260922)
    train = [make_example(rng) for _ in range(1200)]
    test = [make_example(random.Random(99000 + i)) for i in range(260)]

    hard_rows = []
    for i, (state, intent, urgency, churn) in enumerate(HARD):
        hard_rows.append({
            "state": state,
            "questions": QUESTIONS,
            "gold": gold_for(intent, urgency, churn),
            "meta": {"intent": intent, "urgency": urgency, "churn": churn, "kind": "hard"},
        })

    write_jsonl(os.path.join(DATA, "cn_train.jsonl"), train)
    write_jsonl(os.path.join(DATA, "cn_test.jsonl"), test)
    write_jsonl(os.path.join(DATA, "cn_test_hard.jsonl"), hard_rows)

    # 标签分布 sanity check
    from collections import Counter
    for name, rows in [("train", train), ("test", test), ("hard", hard_rows)]:
        ci = Counter(r["meta"]["intent"] for r in rows)
        cu = Counter(r["meta"]["urgency"] for r in rows)
        cc = Counter(str(r["meta"]["churn"]) for r in rows)
        print(name, "intent:", dict(ci), "urgency:", dict(sorted(cu.items())), "churn:", dict(cc))


if __name__ == "__main__":
    main()
