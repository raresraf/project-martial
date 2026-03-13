/**
 * @file 1343C.CPP
 * @brief Solution for CodeForces Problem 1343C - "Alternating Subsequence".
 *
 * @details
 * The problem is to find a subsequence with the maximum possible sum, with the constraint
 * that the signs of adjacent elements must alternate (positive, negative, positive, ... or vice-versa).
 *
 * @algorithm
 * The problem can be solved greedily. The input array can be partitioned into contiguous
 * blocks of numbers with the same sign. For example, in `[1, 2, -3, -4, 5]`, the blocks
 * are `[1, 2]`, `[-3, -4]`, and `[5]`.
 *
 * To maximize the sum, from each block, we should pick only the largest number. In the
 * example, we would pick `2` from the first block, `-3` from the second, and `5` from the third.
 * The final answer is the sum of these chosen numbers: `2 + (-3) + 5 = 4`.
 *
 * This code implements this logic by iterating through the array and keeping track of the
 * maximum element (`last`) in the current block of same-signed numbers. When the sign changes,
 * the `last` element is added to the total sum (`ans`), and the process repeats for the new block.
 *
 * @time_complexity O(N) for each test case, as we iterate through the array once.
 * @space_complexity O(1), as we only store a few variables.
 */
#include <bits/stdc++.h>
#define ll long long
using namespace std;

/**
 * @brief Checks if two numbers have different signs.
 * @param a The first number.
 * @param b The second number.
 * @return True if signs are different, false otherwise.
 *
 * @logic This works because if signs are the same, `|a-b| == ||a|-|b||`.
 *        If signs are different, `|a-b| == |a|+|b|`, which is not equal to `||a|-|b||` (unless one is zero).
 */
bool dif(ll a, ll b) {
    // A more readable way to write this would be: `(a > 0 && b < 0) || (a < 0 && b > 0)`
    return (a * b) < 0;
}

void solve() {
    int n;
    scanf(" %d", &n);
    ll last = 0; // Stores the maximum element of the current block.
    ll ans = 0;  // Stores the sum of maximums from completed blocks.

    /**
     * @brief Main loop to process the array.
     * This loop identifies blocks of same-signed numbers and finds the max of each.
     */
    for (int i = 0; i < n; ++i) {
        ll va;
        scanf(" %lld", &va);

        // A new block begins if the sign of `va` is different from `last`, or if it's the first element.
        if (i == 0) {
            last = va;
        } else if (dif(last, va)) {
            // The block has ended. Add its max (`last`) to the answer.
            ans += last;
            // Start a new block with the current element.
            last = va;
        } else {
            // Still in the same block. Update `last` to be the max element seen so far.
            last = max(last, va);
        }
    }
    // After the loop, the maximum of the very last block is still in `last`. Add it to the answer.
    printf("%lld
", ans + last);
}

int main() {
    int t;
    scanf(" %d", &t);
    while (t--) {
        solve();
    }
    return 0;
}
