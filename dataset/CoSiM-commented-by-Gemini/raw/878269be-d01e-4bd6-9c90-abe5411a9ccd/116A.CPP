/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem using C-style I/O.
 *
 * @details
 * This program calculates the minimum capacity required for a tram by tracking
 * the number of passengers over a series of stops. The capacity is determined
 * by the peak number of passengers on the tram at any given time.
 *
 * Algorithm:
 * 1. Read the total number of stops, n.
 * 2. Initialize `sum` (current passengers) and `max` (peak passengers) to 0.
 * 3. Loop n times, for each stop:
 *    a. Read the number of passengers exiting (`a`) and entering (`b`).
 *    b. Update the current passenger count: `sum = sum - a + b`.
 *    c. If the new `sum` is greater than `max`, update `max` to the new `sum`.
 * 4. After the loop, `max` holds the minimum required capacity. Print `max`.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1).
 */
#include <stdio.h>
int main() {
  int n, a, b, sum = 0, max = 0;
  // n: number of stops
  // a: passengers exiting, b: passengers entering
  // sum: current number of passengers
  // max: peak number of passengers
  scanf("%d", &n);

  /**
   * @brief Loop to simulate the tram's state at each stop.
   */
  for (int i = 1; i <= n; i++) {
    scanf("%d %d", &a, &b);
    // Update the current passenger count: subtract exiting, add entering.
    sum = sum - a;
    sum = sum + b;

    // If the current count is a new maximum, update the peak.
    if (sum > max)
      max = sum;
  }
  printf("%d\n", max);
  return 0;
}
