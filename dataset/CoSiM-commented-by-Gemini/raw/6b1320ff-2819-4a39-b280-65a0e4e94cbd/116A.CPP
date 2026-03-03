
#include <bits/stdc++.h>

using namespace std;

/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A, "Tram".
 * @details The program calculates the minimum possible capacity of a tram by
 * finding the maximum number of passengers on the tram at any given time.
 *
 * Algorithm:
 * The program simulates the change in the number of passengers at each tram stop.
 * It reads the number of stops, and for each stop, the number of people exiting
 * and entering. It maintains a running count of the current number of passengers
 * and a separate variable to track the maximum count seen so far. After each
 * stop, the current passenger count is updated, and if this new count is greater
 * than the maximum seen, the maximum is updated.
 *
 * Time Complexity: O(N), where N is the number of stops, due to a single pass
 * to read data and another to process it.
 * Space Complexity: O(N) to store the exit and entry numbers for each stop.
 */
int main() {
  // n: number of tram stops
  // a: array to store passengers exiting at stop i
  // b: array to store passengers entering at stop i
  // c: current number of passengers on the tram
  // max: maximum number of passengers seen so far
  int n, i, a[1002], b[1002], c;
  long int max;

  scanf("%d", &n);
  // Read the exit and entry data for all n stops.
  for (i = 1; i <= n; i++) {
    scanf("%d %d", &a[i], &b[i]);
  }

  /**
   * @brief Initialize the state after the first stop.
   * At the first stop, a[1] is always 0. The passengers are b[1].
   * The code calculates `c = a[1] + b[1]`, which correctly resolves to `b[1]`.
   */
  c = a[1] + b[1];
  max = c;

  /**
   * @brief Simulate the tram journey from the second stop to the end.
   *
   * Pre-condition: `c` holds the passenger count after stop 1.
   * Invariant: `max` holds the maximum passenger count up to the current stop.
   *
   * Logic: For each stop `i+1`, the new passenger count is calculated as
   * `current_passengers - exits + entries`. The code implements this as
   * `c = c - (a[i+1] - b[i+1])`, which is equivalent to `c = c - a[i+1] + b[i+1]`.
   */
  for (i = 1; i <= n - 1; i++) {
    c = c - (a[i + 1] - b[i + 1]);
    if (c > max) {
      max = c;
    }
  }

  // Output the minimum required capacity.
  cout << max << endl;
}
