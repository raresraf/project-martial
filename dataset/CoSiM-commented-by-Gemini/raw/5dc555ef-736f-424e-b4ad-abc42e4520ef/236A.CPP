/**
 * @file 236A.CPP
 * @brief Generates a limited number of unique pairs of integers.
 * @details This program reads a number of test cases. For each case, it reads
 * integers `n` and `p`, and then generates up to `2*n + p` unique pairs `(i, j)`
 * where `1 <= i < j <= n`.
 *
 * @note The logic in this file does not solve the Codeforces problem 236A
 * ("Boy or Girl"), which involves counting distinct characters in a string. This
 * code addresses a different, pair-generation-related problem.
 *
 * @algorithm Nested Loop Pair Generation
 * 1. For each test case, calculate a limit `m = 2 * n + p`.
 * 2. Use nested loops to generate all unique pairs `(i, j)` from the range [1, n].
 * 3. A counter `m` is decremented for each pair printed. The loops terminate
 *    prematurely if `m` reaches zero, ensuring that no more than the specified
 *    limit of pairs is printed.
 *
 * @complexity
 * - Time: O(N^2) per test case in the worst case, as it may generate all
 *   N*(N-1)/2 pairs. However, it stops early if the limit `m` is reached.
 * - Space: O(1).
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
using namespace std;

// n and p are problem-specific input variables. Their exact meaning
// depends on the original problem statement.
int n, p;

int main() {
  // i, j: loop counters for generating pairs.
  // cas: number of test cases.
  int i, j, cas;
  scanf("%d", &cas);

  // Process each test case.
  while (cas--) {
    scanf("%d%d", &n, &p);

    // `m` is the maximum number of pairs to be printed for this test case.
    int m = 2 * n + p;

    /**
     * @brief Generate and print unique pairs (i, j) until the limit `m` is reached.
     * The outer loop iterates from i = 1 to n.
     * The inner loop iterates from j = i + 1 to n, ensuring that j > i and each
     * pair is unique.
     * The `&& m` condition in the loops causes them to terminate as soon as `m` is 0.
     */
    for (i = 1; i <= n && m; i++) {
      for (j = i + 1; j <= n && m; j++) {
        // Print the current pair and decrement the remaining pairs counter.
        printf("%d %d\n", i, j);
        m--;
      }
    }
  }
  return 0; // Indicate successful execution.
}
