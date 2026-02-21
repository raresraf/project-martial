/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem (Codeforces 116A).
 *
 * @details This program calculates the minimum possible capacity of a tram. It is given
 * the number of passengers who exit and enter at each stop. The capacity must be
 * large enough to accommodate the maximum number of people on the tram at any given time.
 *
 * @algorithm The program simulates the passenger flow stop by stop.
 * 1. It initializes `in` (current passengers) and `capacity` (max passengers) to 0.
 * 2. It loops `n` times, once for each stop.
 * 3. At each stop, it reads the number of exiting (`a`) and entering (`b`) passengers.
 * 4. It updates the current passenger count: `in = in - a + b`.
 * 5. It updates the required `capacity` to be the maximum of its current value and `in`.
 * 6. The final `capacity` value is the minimum required capacity for the tram.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of tram stops, due to a single pass.
 * Space Complexity: O(1), as only a few variables are needed for the simulation.
 */
#include <stdio.h>
#include <string.h> // string.h is included but not used.
int main() {
  int i, n, a, b, t1 = 0, temp = 0, in = 0; // t1 is an unused variable.
  scanf("%d", &n);

  /**
   * @brief Block Logic: Simulates passenger count over `n` stops.
   * @invariant After `i` iterations, `in` holds the passenger count after the
   * `i`-th stop, and `temp` holds the maximum passenger count observed so far.
   */
  for (i = 0; i < n; i++) {
    scanf("%d %d", &a, &b);
    // Update the number of people currently inside the tram.
    in = b - a + in;
    // Check if the current passenger count exceeds the max capacity seen so far.
    if (i == 0 || in > temp) {
      temp = in;
    }
  }
  // Print the minimum required capacity.
  printf("%d", temp);

  return 0;
}