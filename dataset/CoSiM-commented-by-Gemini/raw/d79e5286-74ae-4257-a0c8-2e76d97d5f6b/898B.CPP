/**
 * @file 898B.CPP
 * @brief Solution for a problem involving a linear Diophantine equation.
 * @details This program determines if there exist non-negative integers x and y
 * that satisfy the equation `a*x + b*y = n`, given n, a, and b. If a solution
 * exists, it prints "YES" followed by the values of x and y. Otherwise, it prints "NO".
 *
 * @author ManuelLoaiza
 *
 * @algorithm
 * The program employs a brute-force search for the value of x (represented as `i` in the code).
 * It iterates through possible values of `i` starting from 0.
 * In each iteration, it checks two conditions:
 * 1. If `i * a` exceeds `n`, it's impossible to find a non-negative `y` because `b*y` would have to be negative.
 *    In this case, no solution exists, and the program terminates.
 * 2. It checks if the remainder `(n - i * a)` is perfectly divisible by `b`. If it is, a valid integer `y`
 *    has been found, where `y = (n - i * a) / b`.
 *
 * If a solution is found, the program prints "YES" and the pair (x, y) and exits.
 * The loop has an upper bound of 10,000,000, assumed to be sufficient for the problem's constraints.
 *
 * @complexity
 * Time Complexity: O(n/a), as the loop runs until `i*a > n`. In the worst case, it's bounded by the loop limit.
 * Space Complexity: O(1), as it uses a constant amount of memory.
 */
//
// Created by ManuelLoaiza on 12/16/17.
//
#include <bits/stdc++.h>

using namespace std;

int main() {

  long long n, a, b;
  cin >> n >> a >> b;

  /**
   * Block Logic: This loop iterates through possible values for the coefficient 'x' (named `i`)
   * to find a solution to the equation a*x + b*y = n.
   * Pre-condition: n, a, and b are read from input.
   * Invariant: `i` is the current candidate value for x.
   */
  for (long long i = 0; i <= 10000000; i++) {

    /**
     * Block Logic: Optimization to terminate early. If `i * a` already exceeds `n`,
     * then `n - i * a` would be negative. Since `b` and `y` must be non-negative,
     * no solution is possible for this `i` or any larger `i`.
     */
    if (i * a > n) {
      cout << "NO" << endl;
      break;
    }

    /**
     * Block Logic: Checks if a valid integer solution for `y` exists for the current `i`.
     * If `(n - i * a)` is divisible by `b`, then we have found a valid pair (x, y).
     */
    if ((n - i * a) % b == 0) {
      cout << "YES" << endl;
      long long x = i;
      long long y = (n - i * a) / b;
      cout << x << " " << y << endl;
      break; // A solution is found, so we exit the loop.
    }
  }

  return 0;
}
