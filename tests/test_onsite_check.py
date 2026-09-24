"""现场装管复核测试: python -m unittest discover -s tests

核心断言: 子集动态规划求得的最小总绝对质量差、最优完全匹配条数、每个实测
槽的候选试管集合, 必须与全排列暴力枚举一致(含等质量互换导致的多解);
候选唯一且为原槽→正确, 候选唯一但属他槽→错槽, 多解→身份不确定;
读数数量与方案管数不等、偏差内无完全匹配时 matchable=false 并给出最大
配对数的对照匹配; 禁用槽/重复槽/非正质量/超 12 管必须拒绝; 复核不修改
入参(只读)。
"""
import copy
import itertools
import random
import unittest

from app.onsite_check import verify_loading


def _plan(placements, radius=100.0, enabled_extra=()):
    """placements: {槽号: (试管编号, 质量, 角度°)}。"""
    slot_state = [
        {"slot": s, "angle": a, "enabled": True, "tube_id": None}
        for s, (_, _, a) in placements.items()
    ]
    for s in enabled_extra:
        slot_state.append({"slot": s, "angle": s * 30, "enabled": True,
                           "tube_id": None})
    return {
        "id": 1,
        "name": "方案1",
        "created_at": "2026-09-24T00:00:00+00:00",
        "request": {"radius_mm": radius},
        "result": {
            "config": {"radius_mm": radius},
            "slot_state": slot_state,
            "placements": [
                {"tube_id": t, "slot": s, "mass": m, "angle": a}
                for s, (t, m, a) in placements.items()
            ],
        },
    }


def _brute(plan, reads, tol):
    """全排列参考: (最小总质量差, 最优条数, {实测槽: 候选管集合}) 或 None(无完全匹配)。"""
    used = [p for p in plan["result"]["placements"] if p.get("slot") is not None]
    n = len(used)
    if len(reads) != n:
        return None
    best = None
    cnt = 0
    cand = {s: set() for s, _ in reads}
    for perm in itertools.permutations(range(n)):
        total = 0.0
        pairs = []
        ok = True
        for i, j in enumerate(perm):
            d = abs(reads[i][1] - used[j]["mass"])
            if d > tol + 1e-9 * max(1.0, tol):
                ok = False
                break
            total += d
            pairs.append((reads[i][0], used[j]["tube_id"]))
        if not ok:
            continue
        if best is None or total < best - 1e-9 * max(1.0, best):
            best, cnt = total, 1
            cand = {s: set() for s, _ in reads}
            for s, tid in pairs:
                cand[s].add(tid)
        elif total <= best + 1e-9 * max(1.0, best):
            cnt += 1
            for s, tid in pairs:
                cand[s].add(tid)
    return best, cnt, cand


def _brute_best_effort(plan, reads, tol):
    """子集枚举参考: (最大配对数, 最小成本)。"""
    used = [p for p in plan["result"]["placements"] if p.get("slot") is not None]
    n = len(used)
    k = len(reads)
    best = None

    def rec(i, used_mask, card, cost):
        nonlocal best
        if i == k:
            if best is None or card > best[0] or (card == best[0] and cost < best[1]):
                best = (card, cost)
            return
        rec(i + 1, used_mask, card, cost)
        for j in range(n):
            bit = 1 << j
            if used_mask & bit:
                continue
            d = abs(reads[i][1] - used[j]["mass"])
            if d <= tol + 1e-9 * max(1.0, tol):
                rec(i + 1, used_mask | bit, card + 1, cost + d)

    rec(0, 0, 0, 0.0)
    return best


class TestLoadingRecheck(unittest.TestCase):
    def test_all_correct(self):
        p = _plan({0: (1, 50.0, 0.0), 6: (2, 51.0, 180.0)})
        r = verify_loading(p, [
            {"slot": 0, "measured_mass_g": 50.02},
            {"slot": 6, "measured_mass_g": 51.01},
        ], 0.05)
        self.assertTrue(r["matchable"])
        self.assertTrue(r["all_correct"])
        self.assertEqual(r["counts"], {"correct": 2, "wrong_slot": 0, "uncertain": 0})
        self.assertAlmostEqual(r["optimal_total_abs_diff_g"], 0.03, places=6)
        self.assertEqual(r["optimal_match_count"], 1)
        for s in r["slots"]:
            self.assertEqual(s["verdict"], "correct")
            self.assertEqual(s["candidate_count"], 1)

    def test_distinct_mass_swap_is_wrong_slot(self):
        # 两管质量不同且整体互换: 最优匹配唯一, 两槽都应判错槽, 不得含糊
        p = _plan({0: (1, 50.0, 0.0), 6: (2, 51.0, 180.0)})
        r = verify_loading(p, [
            {"slot": 0, "measured_mass_g": 51.0},
            {"slot": 6, "measured_mass_g": 50.0},
        ], 0.05)
        self.assertTrue(r["matchable"])
        self.assertEqual(r["counts"], {"correct": 0, "wrong_slot": 2, "uncertain": 0})
        by_slot = {s["slot"]: s for s in r["slots"]}
        self.assertEqual([c["tube_id"] for c in by_slot[0]["candidates"]], [2])
        self.assertEqual([c["tube_id"] for c in by_slot[6]["candidates"]], [1])
        # 代表性匹配如实展示互换
        self.assertEqual(
            {(x["slot"], x["tube_id"]) for x in r["representative_match"]},
            {(0, 2), (6, 1)},
        )
        # 反向汇总: 每支管唯一可能落在对方槽位
        tubes = {t["tube_id"]: t for t in r["tubes"]}
        self.assertEqual(tubes[1]["possible_slots"], [6])
        self.assertEqual(tubes[1]["verdict"], "elsewhere")

    def test_equal_mass_swap_is_uncertain_not_wrong(self):
        # 等质量两管整体互换: 两个完全匹配成本同为 0, 两槽都必须是身份不确定,
        # 不能任取一解断言错放
        p = _plan({0: (1, 50.0, 0.0), 6: (2, 50.0, 180.0)})
        r = verify_loading(p, [
            {"slot": 0, "measured_mass_g": 50.0},
            {"slot": 6, "measured_mass_g": 50.0},
        ], 0.05)
        self.assertTrue(r["matchable"])
        self.assertTrue(r["tie_broken"])
        self.assertEqual(r["optimal_match_count"], 2)
        self.assertEqual(r["counts"], {"correct": 0, "wrong_slot": 0, "uncertain": 2})
        for s in r["slots"]:
            self.assertEqual(s["verdict"], "uncertain")
            self.assertEqual({c["tube_id"] for c in s["candidates"]}, {1, 2})
        self.assertFalse(r["all_correct"])

    def test_mixed_correct_wrong_uncertain(self):
        # 槽 0/6 为等质量对(读数整体互换→各自不确定), 槽 3 质量唯一→正确
        p = _plan({0: (1, 50.0, 0.0), 3: (5, 30.0, 90.0), 6: (2, 50.0, 180.0)})
        r = verify_loading(p, [
            {"slot": 0, "measured_mass_g": 50.0},
            {"slot": 3, "measured_mass_g": 30.0},
            {"slot": 6, "measured_mass_g": 50.0},
        ], 0.05)
        by_slot = {s["slot"]: s for s in r["slots"]}
        self.assertEqual(by_slot[0]["verdict"], "uncertain")
        self.assertEqual(by_slot[6]["verdict"], "uncertain")
        self.assertEqual(by_slot[3]["verdict"], "correct")
        self.assertEqual(r["counts"], {"correct": 1, "wrong_slot": 0, "uncertain": 2})

    def test_no_complete_matching_within_tolerance(self):
        p = _plan({0: (1, 50.0, 0.0), 6: (2, 51.0, 180.0)})
        r = verify_loading(p, [
            {"slot": 0, "measured_mass_g": 80.0},
            {"slot": 6, "measured_mass_g": 81.0},
        ], 0.05)
        self.assertFalse(r["matchable"])
        self.assertIn("不存在一对一完全匹配", r["reason"])
        self.assertEqual(r["best_effort"]["matched_count"], 0)
        self.assertNotIn("slots", r)  # 无完全匹配时不下逐槽判定

    def test_tolerance_boundary_is_inclusive(self):
        p = _plan({0: (1, 50.0, 0.0)})
        r = verify_loading(p, [{"slot": 0, "measured_mass_g": 50.5}], 0.5)
        self.assertTrue(r["matchable"])
        self.assertAlmostEqual(r["optimal_total_abs_diff_g"], 0.5, places=6)
        # 超出偏差则不可匹配
        r2 = verify_loading(p, [{"slot": 0, "measured_mass_g": 50.6}], 0.5)
        self.assertFalse(r2["matchable"])

    def test_reading_count_fewer(self):
        p = _plan({0: (1, 50.0, 0.0), 6: (2, 51.0, 180.0)})
        r = verify_loading(p, [{"slot": 0, "measured_mass_g": 50.0}], 0.05)
        self.assertFalse(r["matchable"])
        self.assertIn("少于方案管数", r["reason"])
        be = r["best_effort"]
        self.assertEqual(be["matched_count"], 1)
        self.assertEqual(len(be["unmatched_tubes"]), 1)
        self.assertEqual(be["unmatched_tubes"][0]["tube_id"], 2)
        self.assertEqual(len(be["unmatched_readings"]), 0)

    def test_reading_count_more_with_empty_slot(self):
        p = _plan({0: (1, 50.0, 0.0), 6: (2, 51.0, 180.0)}, enabled_extra=(3,))
        r = verify_loading(p, [
            {"slot": 0, "measured_mass_g": 50.0},
            {"slot": 3, "measured_mass_g": 50.0},
            {"slot": 6, "measured_mass_g": 51.0},
        ], 0.05)
        self.assertFalse(r["matchable"])
        self.assertIn("多于方案管数", r["reason"])
        self.assertIn("#3", r["reason"])
        self.assertEqual(r["best_effort"]["matched_count"], 2)
        self.assertEqual(len(r["best_effort"]["unmatched_readings"]), 1)

    def test_quantization_to_0_01(self):
        p = _plan({0: (1, 50.0, 0.0)})
        r = verify_loading(p, [{"slot": 0, "measured_mass_g": 50.006}], 0.05)
        self.assertTrue(r["readings"][0]["quantized_to_0_01g"])
        self.assertEqual(r["readings"][0]["measured_mass_g"], 50.01)

    def test_random_against_brute_force(self):
        rng = random.Random(20260924)
        for _ in range(150):
            n = rng.randint(1, 7)
            placements = {}
            for i in range(n):
                tid = i + 1
                mass = rng.choice([10, 10, 10.5, 50, 50, 50.5, 51])
                placements[i] = (tid, mass, i * 30)
            p = _plan(placements)
            used = p["result"]["placements"]
            perm = list(range(n))
            rng.shuffle(perm)
            reads = []
            for i, j in enumerate(perm):
                w = round(used[j]["mass"] + rng.choice([-0.02, 0, 0, 0.01, 0.05]), 2)
                w = max(0.01, w)
                reads.append((i, w))
            tol = rng.choice([0.0, 0.01, 0.06, 0.5])
            r = verify_loading(
                p, [{"slot": s, "measured_mass_g": w} for s, w in reads], tol
            )
            b = _brute(p, reads, tol)
            best, cnt, cand = b
            if best is None:
                self.assertFalse(r["matchable"])
                card, cost = _brute_best_effort(p, reads, tol)
                self.assertEqual(r["best_effort"]["matched_count"], card)
                self.assertAlmostEqual(
                    r["best_effort"]["total_abs_diff_g"], round(cost, 2), places=6
                )
                continue
            self.assertTrue(r["matchable"])
            self.assertAlmostEqual(r["optimal_total_abs_diff_g"], round(best, 2), places=6)
            self.assertEqual(r["optimal_match_count"], cnt)
            for s in r["slots"]:
                self.assertEqual(
                    {c["tube_id"] for c in s["candidates"]}, cand[s["slot"]]
                )
                if s["candidate_count"] == 1:
                    tid = s["candidates"][0]["tube_id"]
                    planned = next(
                        pp["slot"] for pp in used if pp["tube_id"] == tid
                    )
                    expect = "correct" if planned == s["slot"] else "wrong_slot"
                    self.assertEqual(s["verdict"], expect)
                else:
                    self.assertEqual(s["verdict"], "uncertain")

    def test_invalid_inputs(self):
        p = _plan({0: (1, 50.0, 0.0)})
        with self.assertRaisesRegex(ValueError, "至少录入"):
            verify_loading(p, [], 0.05)
        with self.assertRaisesRegex(ValueError, "重复录入"):
            verify_loading(p, [
                {"slot": 0, "measured_mass_g": 50.0},
                {"slot": 0, "measured_mass_g": 50.0},
            ], 0.05)
        with self.assertRaisesRegex(ValueError, "不存在"):
            verify_loading(p, [{"slot": 9, "measured_mass_g": 50.0}], 0.05)
        with self.assertRaisesRegex(ValueError, "禁用槽"):
            p2 = copy.deepcopy(p)
            p2["result"]["slot_state"][0]["enabled"] = False
            verify_loading(p2, [{"slot": 0, "measured_mass_g": 50.0}], 0.05)
        with self.assertRaisesRegex(ValueError, "有限正数"):
            verify_loading(p, [{"slot": 0, "measured_mass_g": 0.0}], 0.05)
        with self.assertRaisesRegex(ValueError, "不能为负"):
            verify_loading(p, [{"slot": 0, "measured_mass_g": 50.0}], -0.01)
        with self.assertRaisesRegex(ValueError, "最多"):
            verify_loading(p, [{"slot": 0, "measured_mass_g": 50.0}] * 13, 0.05)

    def test_more_than_12_plan_tubes_rejected(self):
        placements = {i: (i + 1, 50.0, i * 30) for i in range(13)}
        p = _plan(placements)
        reads = [{"slot": i, "measured_mass_g": 50.0} for i in range(13)]
        with self.assertRaisesRegex(ValueError, "上限 12 管"):
            verify_loading(p, reads, 0.05)

    def test_readonly_does_not_mutate_plan(self):
        p = _plan({0: (1, 50.0, 0.0), 6: (2, 50.0, 180.0)})
        snapshot = copy.deepcopy(p)
        verify_loading(p, [
            {"slot": 0, "measured_mass_g": 50.0},
            {"slot": 6, "measured_mass_g": 50.0},
        ], 0.05)
        self.assertEqual(p, snapshot)


if __name__ == "__main__":
    unittest.main()
