/**
 * @file 898B.CPP
 * @brief Finds a non-negative integer solution to the linear Diophantine equation ax + by = n.
 *
 * @details This program takes three integers n, a, and b as input and attempts to find
 * any pair of non-negative integers (x, y) such that a*x + b*y = n.
 *
 * Algorithm: Iterative search (brute-force on one variable). It iterates through all
 * possible values for x (from 0 to n/a) and checks if the remainder (n - a*x) is
 * divisible by b.
 * Time Complexity: O(N/A), where N is the target sum and A is the first coefficient.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

/**
 * @def optimizar_io
 * @brief A macro to disable C++ stream synchronization with C stdio and untie cin/cout.
 * @details This is a common performance optimization in competitive programming to speed up I/O.
 */
#define optimizar_io                                                           
  ios_base::sync_with_stdio(0);                                                
  cin.tie(0);

using namespace std;

long long idx = 0, x = -1, y, n, a, b;

int main() {
  optimizar_io;
  cin >> n >> a >> b;

  /**
   * Block Logic: Iterates through all possible values for the first variable 'x' (represented by i).
   * Pre-condition: `i*a` must not exceed the total `n`.
   * Invariant: For each `i`, we check if a valid integer `y` exists.
   */
  for (int i = 0; i * a <= n; ++i) {
    // Block Logic: Checks if the remaining amount is perfectly divisible by `b`.
    // If so, we have found a valid integer solution for `y`.
    if ((n - (i * a)) % b == 0) {
      // A valid solution is found. Store it. Any solution will do.
      x = i;
      y = (n - (i * a)) / b;
    }
  }

  // Block Logic: If x remains -1, no solution was found during the iteration.
  if (x == -1)
    cout << "NO
";
  else
    cout << "YES
" << x << " " << y << "
";
}
