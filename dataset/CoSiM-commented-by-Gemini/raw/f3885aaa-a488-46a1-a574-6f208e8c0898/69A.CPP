/**
 * @file 69A.CPP
 * @brief Attempts to solve the "Young Physicist" problem by summing all vector components.
 *
 * This program reads `n` three-dimensional vectors and attempts to determine if
 * their sum is zero. It reads the x, y, and z components of each vector, adds
 * all three components together, and accumulates this sum into a single variable.
 * It then checks if this final accumulated sum is zero.
 *
 * @warning This program contains a logical flaw for solving the "Young Physicist"
 * problem. Equilibrium requires the sum of forces on each axis (X, Y, and Z)
 * to be zero independently. This code incorrectly combines all components into a
 * single sum, which can lead to false positives (e.g., vectors (1, -1, 0) and
 * (-1, 1, 0) would result in a total sum of 0, but the body is not in equilibrium).
 *
 * @author f3885aaa-a488-46a1-a574-6f208e8c0898
 *
 * @algorithm Flawed Single-Variable Summation
 * @complexity
 * Time: O(N), where N is the number of vectors.
 * Space: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // `n`: number of vectors. `ans`: single accumulator for all components.
  int n, ans = 0;
  scanf("%d", &n);

  /**
   * @brief Loop to read vectors and accumulate component sums incorrectly.
   */
  while (n--) {
    int a, b, c; // Components of a single vector.
    scanf("%d %d %d", &a, &b, &c);
    // Flawed logic: adds all components together instead of tracking axis sums.
    ans += a + b + c;
  }
  
  /**
   * @brief Checks if the single combined sum is zero.
   * This condition is not sufficient to prove vector equilibrium.
   */
  if (ans == 0)
    printf("YES
");
  else
    printf("NO
");
  return 0;
}