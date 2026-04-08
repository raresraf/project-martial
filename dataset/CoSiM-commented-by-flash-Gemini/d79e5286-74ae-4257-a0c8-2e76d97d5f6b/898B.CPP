//
// Created by ManuelLoaiza on 12/16/17.
//
/**
 * @file 898B.CPP
 * @brief Solves the "Proper Nutrition" problem from Codeforces (Problem 898B).
 * @details This program solves a linear Diophantine equation of the form
 * `x*a + y*b = n`, where `x` and `y` must be non-negative integers. It finds
 * if it's possible to obtain a total of `n` by combining items with costs
 * `a` and `b`.
 *
 * @algorithm
 * The solution uses a brute-force search. It iterates through the possible
 * number of items of cost `a` (let's call this quantity `x`).
 * 1. Read the target value `n`, and the costs `a` and `b`.
 * 2. Loop with `x` from 0 upwards.
 * 3. In each iteration, calculate the remaining value needed: `rem = n - x*a`.
 * 4. Check if `rem` is non-negative and if it is perfectly divisible by `b`.
 * 5. If it is, a solution is found (`y = rem / b`). Print "YES" along with
 *    the quantities `x` and `y`, then terminate.
 * 6. If `x*a` exceeds `n`, no solution is possible with the current `x` or any
 *    larger `x`, so print "NO" and terminate.
 *
 * @time_complexity O(N/A), where N is the target value and A is the smaller of the two costs.
 * In this code, it's bounded by the loop limit, effectively O(N).
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  long long n, a, b;
  cin >> n >> a >> b;

  /**
   * @block_logic Iterate through possible quantities `i` for the item with cost `a`.
   */
  for (long long i = 0; i <= 10000000; i++) {

    // If the cost of `i` items of `a` already exceeds `n`, no solution is possible.
    if (i * a > n) {
      cout << "NO" << endl;
      break;
    }

    // Check if the remaining amount `(n - i * a)` can be formed using items of cost `b`.
    if ((n - i * a) % b == 0) {
      cout << "YES" << endl;
      long long x = i;
      long long y = (n - i * a) / b;
      cout << x << " " << y << endl;
      break; // A solution has been found, so we can exit.
    }
  }

  return 0;
}
