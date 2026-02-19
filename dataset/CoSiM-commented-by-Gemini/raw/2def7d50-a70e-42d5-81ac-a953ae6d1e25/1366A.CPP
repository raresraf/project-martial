/**
 * @file 1366A.CPP
 * @brief Solution to the "Shovels and Swords" problem.
 * @details This program calculates the maximum number of items (shovels or swords) that can be
 * crafted given a number of sticks `a` and diamonds `b`. A shovel requires 2 sticks and 1 diamond,
 * while a sword requires 1 stick and 2 diamonds.
 *
 * Algorithm Explanation:
 * To make any single item (either a shovel or a sword), you need at least 1 stick and 1 diamond.
 * Let `k` be the number of items we want to make.
 * The total number of sticks required will be between `k` (if all are swords) and `2k` (if all are shovels).
 * The total number of diamonds required will be between `k` (if all are shovels) and `2k` (if all are swords).
 * The total number of components (sticks + diamonds) needed for `k` items is always `3k`.
 *
 * From this, we have three constraints on `k`:
 * 1. `k <= a` (we can't make more items than we have sticks)
 * 2. `k <= b` (we can't make more items than we have diamonds)
 * 3. `3*k <= a + b` => `k <= (a + b) / 3` (we can't make more items than our total components allow)
 *
 * Combining these, the maximum number of items `k` is `min(a, b, (a + b) / 3)`.
 *
 * The implemented code arrives at this conclusion through a series of checks:
 * 1. It handles some base cases (e.g., if a or b is 0).
 * 2. It ensures `a` is the larger value and `b` is the smaller one.
 * 3. It calculates a theoretical maximum `x = (a + b) / 3`.
 * 4. It then correctly identifies that the answer cannot exceed the minimum of the two resources, `b`.
 * 5. Therefore, the answer is `min(x, b)`, which is equivalent to `min((a + b) / 3, min(a, b))`.
 *
 * Time Complexity: O(1) for each test case.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  scanf("%d", &n);
  while (n--) {
    long long int a, b, x, temp;
    scanf("%lld %lld", &a, &b);

    // The following `if-else` blocks handle edge cases that are also covered by the general solution.
    if (a == 0 || b == 0 || (a == 1 && b == 1)) {
      printf("0\n");
    } else if ((a == 1 && b > 1) || (b == 1 && a > 1)) {
      // With 1 stick and >=2 diamonds, you can make 1 sword.
      // With 1 diamond and >=2 sticks, you can make 1 shovel.
      printf("1\n");
    } else {
      // General case logic.
      // Ensure 'a' holds the larger value and 'b' the smaller one.
      if (b > a) {
        temp = a;
        a = b;
        b = temp;
      }
      
      // The total number of items is limited by the total number of components.
      x = (a + b) / 3;
      
      // The number of items is also limited by the scarcer of the two resources.
      if (x > b) {
        // If the theoretical max `x` is more than our scarcest resource `b`, we are limited by `b`.
        printf("%lld\n", b);
      } else {
        // Otherwise, we are limited by the total number of components.
        printf("%lld\n", x);
      }
    }
  }
  return 0;
}
