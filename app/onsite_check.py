"""已保存配平方案的现场装管复核(只读, 不改变求解与保存格式)。

问题模型
--------
方案在槽位 s 上规定放置标称质量 m 的试管 tube(s)。现场装管完成后, 操作员
按**启用槽**逐槽录入: 哪些槽位实际上有管、以及该管的实测质量 w(精度
0.01 g), 并给出统一允许偏差 Δ(g): 仅当实测质量与某管标称质量之差的绝对
值不超过 Δ 时, 才认为二者可能是同一支管(在这样的边上才允许匹配)。

本模块只在"质量差不超过偏差"的边上做**一对一**匹配(二分图匹配), 并在
所有合法完全匹配中取**总绝对质量差最小**者; 由于方案与读数都可能存在
等质量(或质量差落在同一最优水平上)的试管, 最优匹配往往不止一种。为避免
"任取一个最优解就断言错放", 本模块**枚举汇总全部最优匹配**:

- 对每个实测槽 s, 收集在**任意一个**最优完全匹配中可能匹配到该槽的全部
  试管 C(s);
- C(s) 唯一时: 该试管就是该槽管的确定身份 —— 其方案槽位恰为 s 则判
  **正确**, 否则判**错槽**(即确定是另一方案槽位的管放到了 s);
- C(s) 含多个试管时判**身份不确定**: 存在多个同为最优的一对一匹配,
  无法在不引入额外信息(编号标签等)的情况下确定管的身份, 此时**不得**
  仅凭任一最优解断言错放。

精确算法(位掩码子集动态规划)
-----------------------------
读数 k 个、方案管 n 支(k, n ≤ 12)。把实测槽按槽号升序处理, 方案管按管号
升序编号为 0..n-1; 边 (i,j) 的成本 c(i,j)=|w_i-m_j|, 不可匹配(质量差
超过偏差)的边成本为 +∞。

k == n 时, 逐槽强制分配:

    f[i][mask] = 处理完前 i 个实测槽、已用掉方案管集合 mask 时的最小
                 总成本(每个槽恰分配一支管)                       O(k·2ⁿ)
    g[i][mask] = 同上成本下的匹配条数
    最优总成本 V = f[k][全集]; g[k][全集] 即最优完全匹配条数。

边 (i,j) 属于某个最优完全匹配, 当且仅当存在 mask(含 j、|mask|=i+1) 使

    f[i][mask\\{j}] + c(i,j) + h[i+1][mask] == V

其中 h[i][mask] 是后段 DP: 固定前 i 槽已用集合 mask 时, 把剩余方案管
恰好分配给剩余实测槽的最小成本(强制一一对应, 分配不完则 +∞)。

k != n 或不存在合法完全匹配时, 本模块改求**最大基数**的最小成本匹配作为
对照(先最大化配对数, 再最小化总质量差), 并在 reason 中说明原因,
不对任何槽位做正确/错槽判定。

拒绝条件(ValueError → HTTP 400)
--------------------------------
- 方案没有已放置试管, 或已放置试管超过 12 支;
- 方案摆放明细中试管/槽位重复、质量非正、缺少槽位角度;
- 读数为空、超过 12 条, 实测槽位重复、不存在或为方案禁用槽;
- 实测质量不是有限正数;
- 统一允许偏差不是有限非负数。

读数数量与方案管数不等、或不存在合法完全匹配时不抛错, 而是在返回中以
``matchable=False`` 明确说明原因。
"""
from __future__ import annotations

import math
from typing import Any, Optional

# 现场装管复核的规模上限: 至多 12 管(2^12 = 4096 个 DP 状态)
MAX_READINGS = 12
# 实测质量录入/显示精度
MASS_QUANTUM = 0.01
# 与 app/compare.py / app/loading.py / app/evacuation.py 一致的数值容差
_NUM_TOL = 1e-9


def _round2(x: float) -> tuple[float, bool]:
    """把质量按 0.01 g 量化, 返回(量化值, 是否发生了截断)。"""
    q = round(round(x / MASS_QUANTUM) * MASS_QUANTUM, 2)
    return q, abs(q - x) > 1e-12


def verify_loading(plan: dict[str, Any],
                   readings: list[dict[str, Any]],
                   tolerance_g: float) -> dict[str, Any]:
    """对已保存方案做现场装管复核(只读)。

    参数
    ----
    plan: GET /api/plan/{id} 返回结构的方案(含 id/name/request/result),
        仅读取, 不修改。
    readings: [{"slot": 实测槽位, "measured_mass_g": 实测质量}, ...],
        每项表示该启用槽现场实际有管; 至多 12 项、槽位不重复。
    tolerance_g: 统一允许偏差(g), 非负; 只在 |实测-标称| ≤ 偏差 的边上匹配。

    返回
    ----
    复核结果字典; 数量不一致或无完全匹配时 ``matchable=False`` 并给出
    reason, 完全匹配时逐实测槽给出全部最优匹配候选试管与
    正确/错槽/身份不确定判定。
    """
    plan_id = plan.get("id")

    # ---- 统一允许偏差 ----
    tol = float(tolerance_g)
    if not math.isfinite(tol):
        raise ValueError("统一允许偏差必须为有限数值。")
    if tol < 0.0:
        raise ValueError("统一允许偏差不能为负。")
    tol_tol = _NUM_TOL * max(1.0, tol)

    # ---- 方案摆放明细 ----
    result = plan.get("result") or {}
    used = [p for p in (result.get("placements") or [])
            if p.get("slot") is not None]
    if not used:
        raise ValueError(f"方案 {plan_id} 没有已放置的试管, 无法进行现场装管复核。")
    if len(used) > MAX_READINGS:
        raise ValueError(
            f"方案 {plan_id} 有 {len(used)} 支已放置试管, 超过现场装管复核"
            f"上限 {MAX_READINGS} 管(子集动态规划状态数为 2^n), 明确拒绝。"
        )

    slot_state = result.get("slot_state") or []
    slot_enabled: dict[int, bool] = {}
    slot_angle: dict[int, Optional[float]] = {}
    for item in slot_state:
        sidx = int(item["slot"])
        slot_enabled[sidx] = bool(item.get("enabled", True))
        slot_angle[sidx] = float(item["angle"]) if item.get("angle") is not None else None

    tubes: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_slots: set[int] = set()
    for p in used:
        tid = int(p["tube_id"])
        if tid in seen_ids:
            raise ValueError(f"摆放明细中试管 {tid} 重复。")
        seen_ids.add(tid)
        slot = int(p["slot"])
        if slot in seen_slots:
            raise ValueError(f"摆放明细中槽位 {slot} 被重复占用。")
        seen_slots.add(slot)
        mass = float(p["mass"])
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"试管 {tid} 的标称质量必须为正数。")
        angle = p.get("angle")
        if angle is None:
            raise ValueError(f"试管 {tid} 的摆放结果缺少槽位角度。")
        if slot in slot_enabled and not slot_enabled[slot]:
            raise ValueError(
                f"方案摆放明细中试管 {tid} 落在禁用槽 {slot} 上, 方案数据不一致。"
            )
        tubes.append({
            "tube_id": tid,
            "slot": slot,
            "angle_deg": float(angle),
            "mass": mass,
        })
    tubes.sort(key=lambda t: t["tube_id"])
    n = len(tubes)
    plan_slot_of = {t["tube_id"]: t["slot"] for t in tubes}
    all_plan_slots = sorted(plan_slot_of.values())

    # ---- 现场读数 ----
    if not isinstance(readings, (list, tuple)) or not readings:
        raise ValueError("请至少录入一个实际有管的槽位读数。")
    if len(readings) > MAX_READINGS:
        raise ValueError(
            f"现场读数最多 {MAX_READINGS} 管(收到 {len(readings)} 条)。"
        )
    raw_rows: list[dict[str, Any]] = []
    seen_read_slots: set[int] = set()
    for item in readings:
        slot = int(item["slot"])
        if slot in seen_read_slots:
            raise ValueError(f"实测槽位 {slot} 重复录入。")
        seen_read_slots.add(slot)
        if slot not in slot_enabled:
            raise ValueError(f"实测槽位 {slot} 在方案中不存在。")
        if not slot_enabled[slot]:
            raise ValueError(
                f"实测槽位 {slot} 在方案中为禁用槽, 现场装管复核只接受启用槽。"
            )
        mass = float(item["measured_mass_g"])
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"槽位 {slot} 的实测质量必须为有限正数。")
        mass_q, truncated = _round2(mass)
        raw_rows.append({
            "slot": slot,
            "measured_mass_g": mass_q,
            "truncated": truncated,
            "angle_deg": slot_angle.get(slot),
        })
    raw_rows.sort(key=lambda r: r["slot"])
    k = len(raw_rows)

    # ---- 成本矩阵 c[i][j](不可匹配为 +∞) ----
    INF = math.inf
    cost: list[list[float]] = [[INF] * n for _ in range(k)]
    for i, row in enumerate(raw_rows):
        for j, t in enumerate(tubes):
            d = abs(row["measured_mass_g"] - t["mass"])
            if d <= tol + tol_tol:
                cost[i][j] = d

    response: dict[str, Any] = {
        "method": (
            "位掩码子集动态规划(2^n, n≤12)在质量差不超过统一允许偏差的边上"
            "做一对一匹配, 取总绝对质量差最小的完全匹配; 枚举全部最优匹配"
            "并对每个实测槽汇总可能对应的试管(唯一→正确/错槽, 多解→身份不确定)"
        ),
        "tolerance_g": tol,
        "plan_tube_count": n,
        "reading_count": k,
        "mass_precision_g": MASS_QUANTUM,
        "plan_slots": all_plan_slots,
        "plan": {
            "id": plan.get("id"),
            "name": plan.get("name"),
            "created_at": plan.get("created_at"),
        },
        "readings": [
            {
                "slot": r["slot"],
                "angle_deg": r["angle_deg"],
                "measured_mass_g": r["measured_mass_g"],
                "quantized_to_0_01g": r["truncated"],
            }
            for r in raw_rows
        ],
        "note": "现场装管复核为只读结果, 未修改任何已保存方案。",
    }

    # ---- 数量不一致: 求最大基数最小成本匹配作对照, 不做身份判定 ----
    if k != n:
        response.update(_best_effort(raw_rows, tubes, cost))
        extra = sorted(seen_read_slots - set(all_plan_slots))
        if k < n:
            reason = (
                f"读数数量({k})少于方案管数({n}), 不存在覆盖全部方案试管的"
                "一对一完全匹配; 下方给出配对数最多、总质量差最小的对照匹配,"
                "不能据此对任何槽位下正确/错槽结论。"
            )
        else:
            reason = (
                f"读数数量({k})多于方案管数({n}), 一对一匹配容纳不下全部读数,"
                "不存在完全匹配; 下方给出配对数最多、总质量差最小的对照匹配,"
                "不能据此对任何槽位下正确/错槽结论。"
            )
        if extra:
            reason += (
                " 其中读数槽 "
                + "、".join(f"#{s}" for s in extra)
                + " 是方案中无管的启用槽(多装)。"
            )
        response["matchable"] = False
        response["reason"] = reason
        return response

    # ---- k == n: 前段 DP(逐槽强制分配, 分配不下去为不可达) ----
    size = 1 << n
    full = size - 1
    f = [INF] * ((k + 1) * size)
    g = [0] * ((k + 1) * size)
    f[0] = 0.0
    g[0] = 1

    for i in range(k):
        base_in = i * size
        base_out = (i + 1) * size
        ci = cost[i]
        for mask in range(size):
            cur = f[base_in + mask]
            if cur == INF:
                continue
            ways = g[base_in + mask]
            bits = full ^ mask
            while bits:
                lsb = bits & (-bits)
                j = lsb.bit_length() - 1
                cij = ci[j]
                if cij != INF:
                    nm = mask | lsb
                    idx = base_out + nm
                    val = cur + cij
                    tol_v = _NUM_TOL * max(1.0, f[idx] if f[idx] != INF else val)
                    if f[idx] == INF or val < f[idx] - tol_v:
                        f[idx] = val
                        g[idx] = ways
                    elif val <= f[idx] + tol_v:
                        g[idx] += ways
                bits ^= lsb

    optimum_idx = k * size + full
    if f[optimum_idx] == INF:
        response.update(_best_effort(raw_rows, tubes, cost))
        response["matchable"] = False
        response["reason"] = (
            f"读数数量({k})与方案管数({n})相等, 但在统一允许偏差 "
            f"{tol:g} g 内不存在一对一完全匹配(至少一支实测管与所有方案管的"
            "质量差都超过偏差)。请核对实测质量录入、允许偏差取值, 并检查是否"
            "存在非本方案的试管; 下方给出配对数最多、总质量差最小的对照匹配,"
            "不能据此断言具体错放。"
        )
        return response

    optimum = f[optimum_idx]
    opt_tol = _NUM_TOL * max(1.0, optimum)

    # ---- 后段 DP: h[i][mask] 固定前 i 槽用掉 mask 后, 剩余部分的最小成本 ----
    h = [INF] * ((k + 1) * size)
    h[k * size + full] = 0.0
    for i in range(k - 1, -1, -1):
        base_next = (i + 1) * size
        base_cur = i * size
        ci = cost[i]
        for mask in range(size):
            if mask.bit_count() != i:
                continue
            best = INF
            bits = full ^ mask
            while bits:
                lsb = bits & (-bits)
                j = lsb.bit_length() - 1
                cij = ci[j]
                tail = h[base_next + (mask | lsb)]
                if cij != INF and tail != INF:
                    val = cij + tail
                    if val < best:
                        best = val
                bits ^= lsb
            h[base_cur + mask] = best

    # ---- 每条边是否可能出现在某个最优完全匹配中 ----
    # 边 (i,j) 可能 ⇔ 存在 mask(含 j、|mask|=i+1) 使
    # f[i][mask\j] + c(i,j) + h[i+1][mask] == optimum。
    # 内层总枚举量 Σ_{i,j} C(n-1,i) = n·2^(n-1), n=12 时约 2.5 万次有效检查
    # (含 popcount 过滤的裸循环约 3·10^7 次, 毫秒级)。
    possible: list[list[bool]] = [[False] * n for _ in range(k)]
    for i in range(k):
        ci = cost[i]
        f_base = i * size
        h_base = (i + 1) * size
        need_cnt = i + 1
        for j in range(n):
            cij = ci[j]
            if cij == INF:
                continue
            jbit = 1 << j
            for mask in range(size):
                if not (mask & jbit) or mask.bit_count() != need_cnt:
                    continue
                pre = f[f_base + (mask ^ jbit)]
                tail = h[h_base + mask]
                if pre != INF and tail != INF and pre + cij + tail <= optimum + opt_tol:
                    possible[i][j] = True
                    break

    # ---- 一个代表性最优完全匹配(仅展示; 身份结论以候选集合为准) ----
    rep: list[int] = []
    mask = 0
    for i in range(k):
        chosen: Optional[int] = None
        bits = full ^ mask
        while bits:
            lsb = bits & (-bits)
            j = lsb.bit_length() - 1
            tail = h[(i + 1) * size + (mask | lsb)]
            if (cost[i][j] != INF and tail != INF
                    and f[i * size + mask] + cost[i][j] + tail <= optimum + opt_tol):
                chosen = j
                break
            bits ^= lsb
        if chosen is None:  # pragma: no cover - 与上面可行性矛盾时不发生
            raise AssertionError("代表性最优匹配构造失败: DP 状态不一致。")
        rep.append(chosen)
        mask |= 1 << chosen

    def _pair(row: dict[str, Any], j: int) -> dict[str, Any]:
        t = tubes[j]
        return {
            "slot": row["slot"],
            "angle_deg": row["angle_deg"],
            "tube_id": t["tube_id"],
            "planned_slot": t["slot"],
            "planned_angle_deg": t["angle_deg"],
            "nominal_mass_g": t["mass"],
            "measured_mass_g": row["measured_mass_g"],
            "mass_diff_g": round(row["measured_mass_g"] - t["mass"], 2),
            "abs_mass_diff_g": round(abs(row["measured_mass_g"] - t["mass"]), 2),
        }

    rep_pairs = [_pair(raw_rows[i], j) for i, j in enumerate(rep)]

    # ---- 逐实测槽汇总候选并判定; 同时反向汇总每支方案管的可能落槽 ----
    slot_results: list[dict[str, Any]] = []
    n_correct = n_wrong = n_uncertain = 0
    tube_possible_slots: dict[int, list[int]] = {t["tube_id"]: [] for t in tubes}
    for i, row in enumerate(raw_rows):
        cands = []
        for j in range(n):
            if not possible[i][j]:
                continue
            t = tubes[j]
            cands.append({
                "tube_id": t["tube_id"],
                "planned_slot": t["slot"],
                "nominal_mass_g": t["mass"],
                "abs_mass_diff_g": round(abs(row["measured_mass_g"] - t["mass"]), 2),
            })
            tube_possible_slots[t["tube_id"]].append(row["slot"])
        cands.sort(key=lambda c: c["tube_id"])
        if len(cands) == 1:
            verdict = "correct" if cands[0]["planned_slot"] == row["slot"] else "wrong_slot"
            if verdict == "correct":
                n_correct += 1
            else:
                n_wrong += 1
        else:
            verdict = "uncertain"
            n_uncertain += 1
        slot_results.append({
            "slot": row["slot"],
            "angle_deg": row["angle_deg"],
            "measured_mass_g": row["measured_mass_g"],
            "candidate_count": len(cands),
            "candidates": cands,
            "verdict": verdict,
        })

    tube_results = []
    for t in tubes:
        slots_here = sorted(tube_possible_slots[t["tube_id"]])
        if len(slots_here) == 1 and slots_here[0] == t["slot"]:
            tv = "at_planned_slot"
        elif len(slots_here) == 1:
            tv = "elsewhere"
        else:
            tv = "uncertain"
        tube_results.append({
            "tube_id": t["tube_id"],
            "planned_slot": t["slot"],
            "possible_slots": slots_here,
            "verdict": tv,
        })

    if n_wrong == 0 and n_uncertain == 0:
        summary = (
            f"全部 {n} 个实测槽均唯一匹配到其原方案槽位上的试管, 现场装管"
            "与方案一致(正确)。"
        )
    elif n_wrong == 0:
        summary = (
            f"{n_correct} 个槽位可确认正确, 另有 {n_uncertain} 个槽位身份不确定"
            "(存在多个总质量差同为最小的一对一匹配); 不能仅凭任一最优解断言"
            "错放, 需结合管号标签等额外信息人工复核。"
        )
    else:
        summary = (
            f"{n_correct} 个槽位确认正确, {n_wrong} 个槽位可唯一确定为错槽"
            "(实测管唯一对应到其他方案槽位的试管), "
            f"{n_uncertain} 个槽位身份不确定; 对身份不确定的槽位不得任取一个"
            "最优匹配就断言错放。"
        )

    response.update({
        "matchable": True,
        "optimal_total_abs_diff_g": round(optimum, 2),
        "optimal_match_count": g[optimum_idx],
        "tie_broken": g[optimum_idx] > 1,
        "slots": slot_results,
        "tubes": tube_results,
        "representative_match": rep_pairs,
        "representative_warning": (
            "下列代表性匹配只展示一个达到最小总质量差的一对一方案; 身份结论"
            "以逐槽候选集合为准, 多解时不得据此断言错放。"
        ),
        "counts": {
            "correct": n_correct,
            "wrong_slot": n_wrong,
            "uncertain": n_uncertain,
        },
        "all_correct": n_wrong == 0 and n_uncertain == 0,
        "summary": summary,
    })
    return response


def _best_effort(raw_rows: list[dict[str, Any]],
                 tubes: list[dict[str, Any]],
                 cost: list[list[float]]) -> dict[str, Any]:
    """数量不一致或无完全匹配时: 最大基数、再最小总质量差的对照匹配。

    二维 0/1 DP: dp[i+1][mask] 来自 dp[i][mask](第 i 槽不匹配)或
    dp[i][mask\\j] + c(i,j)(第 i 槽匹配管 j); 状态值取
    (配对数最大, 总成本最小) 的帕累托最优, 并用 back[i+1][mask] 记录
    来源以便回溯。k, n ≤ 12, 状态数 ≤ 13·4096。
    """
    k = len(raw_rows)
    n = len(tubes)
    size = 1 << n
    full = size - 1
    INF = math.inf
    cards = [[-1] * size for _ in range(k + 1)]
    vals = [[0.0] * size for _ in range(k + 1)]
    # back[i+1][mask] = (prev_mask, tube_j); None 表示第 i 槽未匹配
    back: list[list[Optional[tuple[int, Optional[int]]]]] = [
        [None] * size for _ in range(k + 1)
    ]
    cards[0][0] = 0
    vals[0][0] = 0.0

    for i in range(k):
        ci = cost[i]
        for mask in range(size):
            if cards[i][mask] < 0:
                continue
            oc, ov = cards[i][mask], vals[i][mask]

            def improve(nm: int, nc: int, nv: float,
                        src: tuple[int, Optional[int]]) -> None:
                if (cards[i + 1][nm] < 0 or nc > cards[i + 1][nm]
                        or (nc == cards[i + 1][nm]
                            and nv < vals[i + 1][nm]
                            - _NUM_TOL * max(1.0, vals[i + 1][nm]))):
                    cards[i + 1][nm] = nc
                    vals[i + 1][nm] = nv
                    back[i + 1][nm] = src

            # 第 i 个实测槽不匹配
            improve(mask, oc, ov, (mask, None))
            # 第 i 个实测槽匹配一支未用且允许的方案管
            bits = full ^ mask
            while bits:
                lsb = bits & (-bits)
                j = lsb.bit_length() - 1
                cij = ci[j]
                if cij != INF:
                    improve(mask | lsb, oc + 1, ov + cij, (mask, j))
                bits ^= lsb

    best_mask = 0
    for mask in range(1, size):
        if cards[k][mask] < 0:
            continue
        if (cards[k][mask] > cards[k][best_mask]
                or (cards[k][mask] == cards[k][best_mask]
                    and vals[k][mask] < vals[k][best_mask])):
            best_mask = mask

    def _pair(row: dict[str, Any], j: int) -> dict[str, Any]:
        t = tubes[j]
        return {
            "slot": row["slot"],
            "angle_deg": row["angle_deg"],
            "tube_id": t["tube_id"],
            "planned_slot": t["slot"],
            "planned_angle_deg": t["angle_deg"],
            "nominal_mass_g": t["mass"],
            "measured_mass_g": row["measured_mass_g"],
            "mass_diff_g": round(row["measured_mass_g"] - t["mass"], 2),
            "abs_mass_diff_g": round(abs(row["measured_mass_g"] - t["mass"]), 2),
        }

    pairs: list[dict[str, Any]] = []
    matched_read: set[int] = set()
    matched_tube: set[int] = set()
    mask = best_mask
    for i in range(k, 0, -1):
        prev_mask, j = back[i][mask]
        if j is not None:
            matched_read.add(i - 1)
            matched_tube.add(j)
            pairs.append(_pair(raw_rows[i - 1], j))
        mask = prev_mask
    pairs.sort(key=lambda p: p["slot"])

    unmatched_readings = [
        {
            "slot": raw_rows[i]["slot"],
            "angle_deg": raw_rows[i]["angle_deg"],
            "measured_mass_g": raw_rows[i]["measured_mass_g"],
        }
        for i in range(k) if i not in matched_read
    ]
    unmatched_tubes = [
        {
            "tube_id": tubes[j]["tube_id"],
            "planned_slot": tubes[j]["slot"],
            "nominal_mass_g": tubes[j]["mass"],
        }
        for j in range(n) if j not in matched_tube
    ]

    return {
        "best_effort": {
            "matched_count": cards[k][best_mask],
            "total_abs_diff_g": round(vals[k][best_mask], 2),
            "pairs": pairs,
            "unmatched_readings": unmatched_readings,
            "unmatched_tubes": unmatched_tubes,
            "note": (
                "最大配对数、再取总绝对质量差最小的对照匹配; 不构成完全匹配,"
                "不能作为正确/错槽判定依据。"
            ),
        },
    }
