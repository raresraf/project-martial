/**
 * @file 116A.CPP
 * @brief Calculates the minimum capacity of a tram using C-style I/O.
 *
 * This program solves the "Tram" problem from Codeforces. It determines the
 * minimum required capacity by tracking the number of passengers entering and
 * exiting at each stop and finding the maximum number of passengers on board
 * at any point in time. This version uses `scanf` and `printf` for input/output.
 *
 * @algorithm Simulation / Maximum Tracking
 * @complexity
 * - Time: O(N), where N is the number of stops, due to the single loop.
 * - Space: O(1), as memory usage is constant.
 */
#include <iostream>
#include <stdio.h> // For scanf and printf
using namespace std;

int main() {
  // n: number of stops.
  // a: passengers exiting, b: passengers entering.
  // c: current number of passengers, m: maximum capacity required.
  int n, a, b, c(0), m(0);
  scanf("%d", &n);

  /**
   * @brief Block Logic: Simulates passenger flow for `n` stops.
   * Invariant: After each iteration, `c` holds the current passenger count,
   *            and `m` holds the maximum count observed so far.
   */
  while (n--) {
    scanf("%d%d", &a, &b);
    // Update current passenger count: subtract exiting, add entering.
    c -= a;
    c += b;
    // If the current count is a new maximum, update the capacity requirement.
    if (c > m)
      m = c;
  }
  printf("%d\n", m);
  return 0;
}