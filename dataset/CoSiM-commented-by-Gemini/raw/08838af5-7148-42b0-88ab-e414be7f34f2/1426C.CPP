/**
 * @file 1426C.CPP
 * @brief Solves the "Increase and Copy" problem.
 * @details This program calculates the minimum number of moves to obtain at least 'n'
 * items, starting from one. The available moves are: (1) increment the count by one,
 * and (2) copy all existing items.
 *
 * @see https://codeforces.com/problemset/problem/1426/C
 *
 * Algorithm:
 * The optimal strategy involves a combination of increment and copy moves.
 * First, we perform `k-1` increment moves to get `k` items. This takes `k-1` moves.
 * Then, to get at least `n` items from `k`, we need `ceil(n/k)` groups of `k`. This
 * is achieved with `ceil(n/k) - 1` copy operations.
 * The total number of moves is `f(k) = (k - 1) + (ceil(n/k) - 1)`.
 * This function `f(k)` is minimized when `k` is approximately `sqrt(n)`.
 *
 * This program implements a direct calculation of this minimum by checking values of `k`
 * around `floor(sqrt(n))`.
 * Let `root = floor(sqrt(n))`. The minimum is found by comparing the number of moves for
 * `k = root` and `k = root + 1`.
 *
 * The number of moves for a given `k` is `k - 2 + (n + k - 1) / k` using integer arithmetic.
 * - Case 1: `k = root`. Moves = `root - 2 + ceil(n/root)`.
 * - Case 2: `k = root + 1`. Moves = `root - 1 + ceil(n/(root+1))`.
 *
 * The code simplifies the comparison of these two cases into three conditional branches.
 *
 * Time Complexity: O(T) for T test cases, as each case is solved in O(1) with math operations.
 * Space Complexity: O(1).
 */
#include <cmath>
#include <iostream>

using namespace std;

int main() {
  // x: number of test cases.
  int x;
  cin >> x;
  while (x--) {
    // n: the target number of items.
    int n;
    cin >> n;

    // The optimal number of increments is close to sqrt(n).
    int root = floor(sqrt(n));

    // Let f(k) = (k-1) + ceil(n/k)-1 be the number of moves. We are minimizing f(k).
    // The code checks k = root and k = root + 1 implicitly.

    // Case 1: n is a perfect square or slightly less.
    // The minimum is achieved at k = root.
    // Moves = (root - 1) + (n/root - 1) = 2*root - 2, if n is a perfect square.
    if (root * root >= n) {
      cout << 2 * (root - 1) << endl;
    }
    // Case 2: n is between root*root and root*(root+1).
    // The minimum is achieved at k = root, resulting in 2*root - 1 moves.
    else if ((root + 1) * (root) >= n) {
      cout << 2 * root - 1 << endl;
    }
    // Case 3: n is larger than root*(root+1).
    // The minimum is achieved at k = root+1, resulting in 2*root moves.
    else {
      cout << 2 * root << endl;
    }
  }

  return 0;
}
