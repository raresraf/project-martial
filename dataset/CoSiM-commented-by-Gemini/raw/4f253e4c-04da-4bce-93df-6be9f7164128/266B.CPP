/**
 * @file 266B.CPP
 * @brief An incorrect attempt to solve the "Queue at the School" problem.
 *
 * @details The problem requires simulating a queue of boys ('B') and girls ('G'),
 * where in each time step, any 'B' immediately followed by a 'G' swaps
 * places. This implementation attempts to model that but contains several
 * critical flaws.
 *
 * @algorithm (Correct Algorithm)
 * A correct approach involves iterating `t` times. In each iteration, scan
 * the queue from left to right. If a 'B' is found at index `i` and a 'G' at
 * `i+1`, swap them. To handle simultaneous swaps correctly, the index `i`
 * should be advanced an extra step after a swap.
 *
 * @warning
 * The implementation below is **non-functional and severely flawed**.
 * 1. **Input Reading is incorrect:** It uses `scanf` with "%s" inside a loop,
 *    which causes buffer overflows and does not read the input string correctly.
 * 2. **Algorithm is incorrect:** The simulation loop iterates backward and swaps
 *    `a[i]` with `a[i+1]`, which does not model the problem and leads to
 *    out-of-bounds memory access.
 * 3. **Indexing is unconventional:** The code uses 1-based indexing for C-style
 *    arrays, which is error-prone.
 *
 * The following comments describe the code as written, highlighting its issues.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, t, i, j;
  char a[101]; // C-style array to hold the queue.

  scanf("%d %d", &n, &t);

  /**
   * @block
   * @brief Flawed input reading loop.
   * @warning This loop attempts to read a string using "%s" into sequential
   * character addresses (`&a[i]`), which will corrupt memory and not read the
   * input as intended. The array `a` will not contain the correct initial queue.
   */
  for (i = 1; i <= n; i++) {
    scanf("%s", &a[i]);
  }

  // Outer loop for time steps.
  for (j = 1; j <= t; j++) {
    /**
     * @block
     * @brief Flawed simulation logic for a single time step.
     * @warning This loop iterates backward. The condition `a[i] == 'B'` and the
     * subsequent swap with `a[i+1]` does not match the problem's requirement
     * of swapping a 'B' with a 'G' to its right. It also causes an
     * out-of-bounds access when `i` is `n`. The logic is incorrect.
     */
    for (i = n; i >= 1;) {
      if (a[i] == 'B' && i != n) {
        swap(a[i], a[i + 1]);
        i = i - 2;
      } else {
        i--;
      }
    }
  }

  /**
   * @block
   * @brief Output loop using 1-based indexing.
   * @warning This will print characters from index 1 to n, missing any
   * character at index 0 and potentially reading past the intended string if
   * the input had been read correctly.
   */
  for (i = 1; i <= n; i++) {
    cout << a[i];
  }
  printf("\n");
  return 0;
}