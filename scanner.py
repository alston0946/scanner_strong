import streamlit as st

st.set_page_config(page_title="金首饰利润率计算器", page_icon="💰", layout="wide")

st.title("💰 金首饰利润率计算器")
st.caption("适用于金首饰销售报价、成本核算、利润与利润率计算")

# -----------------------------
# 固定参数
# -----------------------------
SALES_SHIPPING = 15          # 销售运费
COST_SHIPPING = 13           # 运费成本价
SALES_CERT_FEE = 5           # 证书销售费
COST_CERT_FEE = 4            # 证书成本价
COST_LABOR_PER_G = 20        # 成本手工费/克

SILVER_CHAIN_COST = 33
SILVER_CHAIN_SALE = 59

CRAFT_OPTIONS = {
    "珐琅": 105,
    "素金": 95,
    "钻石": 115
}

# -----------------------------
# 输入区
# -----------------------------
st.subheader("基础信息")

col1, col2, col3 = st.columns(3)

with col1:
    weight = st.number_input("克重（g）", min_value=0.0, value=1.0, step=0.01)
    category = st.selectbox("品种", ["珐琅", "素金", "钻石"])

with col2:
    sale_gold_price_per_g = st.number_input("销售金价 / 克（元）", min_value=0.0, value=800.0, step=1.0)
    cost_gold_price_per_g = st.number_input("成本金价 / 克（元）", min_value=0.0, value=700.0, step=1.0)

with col3:
    tax_rate_percent = st.number_input("税率（输入 5 表示 5%）", min_value=0.0, value=5.0, step=0.1)

st.subheader("配件选择")

col4, col5 = st.columns(2)

with col4:
    use_silver_chain = st.checkbox("是否选择银链", value=False)

with col5:
    use_braided_rope = st.checkbox("是否选择编绳", value=False)

braid_cost = 0
braid_sale = 0

if use_braided_rope:
    braid_option = st.radio(
        "请选择编绳档位",
        ["成本15 / 售价30", "成本20 / 售价40"],
        horizontal=True
    )
    if braid_option == "成本15 / 售价30":
        braid_cost = 15
        braid_sale = 30
    else:
        braid_cost = 20
        braid_sale = 40

silver_chain_cost = SILVER_CHAIN_COST if use_silver_chain else 0
silver_chain_sale = SILVER_CHAIN_SALE if use_silver_chain else 0

# -----------------------------
# 计算区
# -----------------------------
tax_rate = tax_rate_percent / 100.0
sale_labor_per_g = CRAFT_OPTIONS[category]

sale_gold_total = sale_gold_price_per_g * weight
cost_gold_total = cost_gold_price_per_g * weight

sale_labor_total = sale_labor_per_g * weight
cost_labor_total = COST_LABOR_PER_G * weight

# 未税销售合计
untaxed_sale_total = (
    sale_gold_total
    + sale_labor_total
    + SALES_SHIPPING
    + SALES_CERT_FEE
    + silver_chain_sale
    + braid_sale
)

# 根据你确认的规则：
# 税费 = 总销售价 × 税率
# 总销售价 = 未税销售合计 + 税费
# => 总销售价 = 未税销售合计 / (1 - 税率)
if tax_rate >= 1:
    st.error("税率不能大于或等于 100%，请重新输入。")
else:
    total_sale_price = untaxed_sale_total / (1 - tax_rate)
    tax_fee = total_sale_price * tax_rate

    # 总成本（按你的业务口径，把同一税费纳入成本展示）
    total_cost = (
        cost_gold_total
        + cost_labor_total
        + COST_SHIPPING
        + COST_CERT_FEE
        + silver_chain_cost
        + braid_cost
        + tax_fee
    )

    profit = total_sale_price - total_cost

    if total_cost == 0:
        profit_rate = 0
    else:
        profit_rate = profit / total_cost

    # -----------------------------
    # 展示区
    # -----------------------------
    st.subheader("计算结果")

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("总销售价", f"¥ {total_sale_price:.2f}")
    top2.metric("总成本", f"¥ {total_cost:.2f}")
    top3.metric("利润", f"¥ {profit:.2f}")
    top4.metric("利润率", f"{profit_rate:.2%}")

    st.subheader("明细拆分")

    left, right = st.columns(2)

    with left:
        st.markdown("### 销售端")
        st.write(f"品种：{category}")
        st.write(f"销售手工费 / 克：¥ {sale_labor_per_g:.2f}")
        st.write(f"销售金价总价：¥ {sale_gold_total:.2f}")
        st.write(f"销售手工费总价：¥ {sale_labor_total:.2f}")
        st.write(f"销售运费：¥ {SALES_SHIPPING:.2f}")
        st.write(f"证书销售费：¥ {SALES_CERT_FEE:.2f}")
        st.write(f"银链售卖价：¥ {silver_chain_sale:.2f}")
        st.write(f"编绳售卖价：¥ {braid_sale:.2f}")
        st.write(f"未税销售合计：¥ {untaxed_sale_total:.2f}")
        st.write(f"税费：¥ {tax_fee:.2f}")
        st.write(f"总销售价：¥ {total_sale_price:.2f}")

    with right:
        st.markdown("### 成本端")
        st.write(f"成本手工费 / 克：¥ {COST_LABOR_PER_G:.2f}")
        st.write(f"成本金价总价：¥ {cost_gold_total:.2f}")
        st.write(f"成本手工费总价：¥ {cost_labor_total:.2f}")
        st.write(f"运费成本价：¥ {COST_SHIPPING:.2f}")
        st.write(f"证书成本价：¥ {COST_CERT_FEE:.2f}")
        st.write(f"银链成本价：¥ {silver_chain_cost:.2f}")
        st.write(f"编绳成本价：¥ {braid_cost:.2f}")
        st.write(f"税费（按你的口径纳入展示）：¥ {tax_fee:.2f}")
        st.write(f"总成本：¥ {total_cost:.2f}")

    st.subheader("当前公式说明")
    st.info(
        "1. 未税销售合计 = 销售金价总额 + 销售手工费总额 + 销售运费 + 证书销售费 + 银链售价（如有） + 编绳售价（如有）\n\n"
        "2. 总销售价 = 未税销售合计 ÷ (1 - 税率)\n\n"
        "3. 税费 = 总销售价 × 税率\n\n"
        "4. 总成本 = 成本金价总额 + 成本手工费总额 + 运费成本价 + 证书成本价 + 银链成本价（如有） + 编绳成本价（如有） + 税费\n\n"
        "5. 利润 = 总销售价 - 总成本\n\n"
        "6. 利润率 = 利润 ÷ 总成本"
    )