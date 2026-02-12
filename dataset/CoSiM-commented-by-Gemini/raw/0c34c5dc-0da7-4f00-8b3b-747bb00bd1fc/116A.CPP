/**
 * @file 116A.cpp
 * @brief Solves the "Tram" problem from Codeforces.
 * @details This program calculates the minimum required capacity of a tram by simulating
 * the passenger count over several stops. It reads the number of stops, and for each stop,
 * it reads the number of passengers exiting and entering. It keeps a running total of the
 * current passengers and tracks the maximum value this total reaches.
 *
 * @author Tico (Thiago Ribeiro)
 * @date 3/17/17
 *
 * @algorithm Simulation to find a running maximum.
 * @complexity
 * Time complexity: O(N), where N is the number of stops, as it iterates through each stop once.
 * Space complexity: O(1).
 */

#include <stdio.h>

int main() {
  // a: passengers exiting, b: passengers entering.
  // v: current number of passengers in the tram.
  // m: maximum number of passengers observed so far (the required capacity).
  // n: the total number of stops.
  int a, b, v = 0, m = 0, n;

  scanf("%d", &n);

  /**
   * @brief Main loop to simulate each stop.
   * @invariant Iterates `n` times, processing one stop in each iteration.
   */
  while (n--) {
    scanf("%d %d", &a, &b);
    // Update the current number of passengers.
    v += b - a;
    // If the current number of passengers exceeds the previous maximum, update the maximum.
    if (v > m)
      m = v;
  }

  // Print the minimum required capacity.
  printf("%d\n", m);

  return 0;
}