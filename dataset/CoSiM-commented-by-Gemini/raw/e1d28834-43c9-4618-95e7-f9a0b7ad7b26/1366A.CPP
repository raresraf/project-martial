/**
 * @file 1366A.CPP
 * @brief Solution to Codeforces problem 1366A - Shovels and Swords.
 * @description This program calculates the maximum number of items (shovels or swords)
 * that can be crafted given a number of sticks `a` and diamonds `b`.
 * Crafting costs:
 * - Shovel: 2 sticks, 1 diamond
 * - Sword:  1 stick, 2 diamonds
 *
 * Algorithm: The solution observes that the total number of items `k` is limited
 * by three constraints: `k <= a` (not enough sticks), `k <= b` (not enough diamonds),
 * and `3k <= a + b` (not enough total materials, since each item costs 3 resources).
 * Therefore, the maximum `k` is `min(a, b, (a+b)/3)`. The code implements a
 * procedural equivalent of this logic. It first greedily crafts items that use
 * the more abundant resource to balance the counts of sticks and diamonds. Then,
 * with balanced resources, it crafts items in pairs (1 shovel + 1 sword, costing
 * 3 sticks and 3 diamonds for 2 items). Finally, it checks if one more item can be
 * made from the remainder.
 *
 * Time Complexity: O(1) for each test case.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
  int t;
  scanf("%d", &t);
  while (t--) {
    ll a, b; // a: number of sticks, b: number of diamonds
    scanf("%lld %lld", &a, &b);

    ll res = 0; // The total number of items crafted.

    /**
     * Block Logic: First, greedily balance the resources.
     * This step makes items that require more of the abundant resource
     * until the counts of sticks and diamonds are as close as possible.
     */
    if (a > b) {
      ll diff = a - b;
      // Number of shovels to make is limited by the number of diamonds (b)
      // or the excess sticks.
      diff = min(b, diff);
      res += diff;
      a -= 2 * diff;
      b -= diff;
    } else if (b > a) {
      ll diff = b - a;
      // Number of swords to make is limited by the number of sticks (a)
      // or the excess diamonds.
      diff = min(a, diff);
      res += diff;
      a -= diff;
      b -= 2 * diff;
    }

    /**
     * Block Logic: After balancing, resources `a` and `b` are nearly equal.
     * Now, craft pairs of (1 shovel + 1 sword). Each pair costs 3 sticks
     * and 3 diamonds and yields 2 items.
     */
    ll l = min(a, b);
    ll amt = l / 3; // Number of pairs we can craft.
    res += amt * 2; // Add the items from these pairs.
    a -= amt * 3;
    b -= amt * 3;

    /**
     * Block Logic: Check if one final item can be made from the leftovers.
     * A shovel (2 sticks, 1 diamond) or a sword (1 stick, 2 diamonds) can be made
     * if there are at least 2 of one resource and 1 of the other.
     */
    if (a && b && a + b >= 3)
      res++;
      
    printf("%lld
", res);
  }
}
