/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem.
 * @details This program calculates the maximum number of 2x1 dominoes that can be placed on an M x N board.
 * This version uses a slightly more complex mathematical expression than necessary.
 *
 * Created by LonelyEnvoy on 2017-7-26.
 * A. Domino piling
 * Keywords: simple math
 *
 * Algorithm:
 * The problem is to tile an M x N board with 2x1 dominoes.
 * The total area of the board is `M * N`. Each domino covers 2 squares.
 * The maximum number of dominoes is therefore `floor((M * N) / 2)`.
 * In integer arithmetic, this is simply `(M * N) / 2`.
 *
 * This implementation explicitly handles cases for even and odd `N`:
 * 1. If `N` is even, the area `M * N` is divisible by 2 if `M` is any integer. The board can be perfectly tiled.
 *    The result is `(M * N) / 2`.
 * 2. If `N` is odd, the code calculates the number of dominoes for the `M x (N-1)` part of the board,
 *    which is `(M * (N-1)) / 2`, and then adds the number of dominoes that can fit in the last row of `M` squares,
 *    which is `M / 2`.
 *
 * While this logic is correct, it is a more verbose way of expressing the simple solution `(m * n) / 2`.
 *
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */

#include <cstdio>
using namespace std;

// Unused competitive programming macros.
#define rep(i, a, n) for (int i = a; i < n; i++)
#define erep(i, a, n) for (int i = a; i <= n; i++)
#define per(i, a, n) for (int i = n - 1; i >= a; i++)

int main() {
  int m, n;
  scanf("%d%d", &m, &n);
  
  /**
   * Block Logic: Calculate the maximum number of dominoes.
   * This ternary operator explicitly handles the cases for 'n' being even or odd.
   * While correct, the single expression `(m * n) / 2` would yield the same result due to integer division.
   */
  int result = (n % 2 == 0) ? (m * n / 2) : ((m * (n - 1)) / 2 + m / 2);
  
  printf("%d\n", result);
}
