/**
 * @file 116A.CPP
 * @brief An efficient C++ solution to the "Tram" problem (Codeforces 116A).
 *
 * @details This program calculates the minimum required capacity of a tram by simulating
 * the passenger count at each stop and tracking the maximum value. This version is
 * implemented in a C-style using `scanf` and `printf` for I/O.
 *
 * @algorithm The program simulates the passenger flow over `n` stops to find the
 * peak occupancy.
 * 1. Initialize `x` (current passengers) and `capacity` (max passengers) to 0.
 * 2. Loop `n` times, once for each stop.
 * 3. At each stop, update the current passenger count: `x = x - a + b`, where `a`
 *    is the number of exiting passengers and `b` is the number of entering ones.
 * 4. If the current count `x` is greater than `capacity`, update `capacity` to `x`.
 * 5. After the loop, `capacity` holds the minimum required capacity.
 *
 * @complexity
 * Time Complexity: O(N), for a single pass through the `N` stops.
 * Space Complexity: O(1), as it only uses a few variables for the simulation.
 */
#include <cstdio>
using namespace std;

int main() {
  int n, a, b, x = 0, capacity = 0;
  scanf("%d", &n);
  
  /**
   * @brief Block Logic: Simulates passenger count over `n` stops.
   * @invariant After each iteration, `x` holds the current number of passengers,
   * and `capacity` holds the maximum number of passengers seen so far.
   */
  while (n--) {
    scanf("%d%d", &a, &b);
    // Update the number of people currently inside the tram.
    x = x - a + b;
    // If current passenger count is the new maximum, update capacity.
    if (x > capacity)
      capacity = x;
  }
  // Print the minimum required capacity.
  printf("%d", capacity);
}
