/**
 * @file 116A.CPP
 * @brief Calculates the maximum capacity required for a tram during its route.
 *   This program simulates a tram journey through 'n' stops, tracking passenger
 *   exits and entries at each stop to determine the maximum number of passengers
 *   present in the tram at any point.
 * * Algorithm: Iterative Capacity Tracking.
 *   The program maintains a running count of passengers in the tram. At each stop,
 *   it updates this count based on exiting and entering passengers and records the
 *   highest count encountered.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(N) for storing arrays 'a', 'b', and 'd'. Could be optimized to O(1)
 *   by not storing all 'a' and 'b' values, and by calculating 'd[i]' directly from 'd[i-1]'
 *   without storing the entire 'd' array.
 */

#include <iostream>

using namespace std;

/**
 * @brief Main function to simulate the tram journey and find maximum capacity.
 * Functional Utility: Handles input, tracks passenger count, and determines the peak capacity.
 */
int main() {
  int n; // Number of tram stops.
  int c = 0; // Stores the maximum capacity encountered.

  cin >> n;

  // Arrays to store passengers exiting (a), entering (b), and current count (d) at each stop.
  int a[n], b[n], d[n];

  /**
   * Block Logic: Reads passenger exit and entry counts for each tram stop.
   * Invariant: After this loop, 'a' and 'b' arrays are populated with respective passenger counts.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }

  // Initializes the passenger count after the first stop.
  d[0] = b[0] - a[0];

  /**
   * Block Logic: Calculates the number of passengers in the tram after each subsequent stop.
   * Invariant: d[i] accurately reflects the number of passengers in the tram after stop 'i'.
   */
  for (int i = 1; i < n; i++) {
    d[i] = d[i - 1] - a[i] + b[i];
  }

  /**
   * Block Logic: Determines the maximum passenger capacity reached during the journey.
   * Invariant: 'c' always holds the highest passenger count observed so far.
   */
  for (int i = 0; i < n; i++) {
    if (d[i] >= c) {
      c = d[i];
    }
  }

  // Outputs the maximum capacity required.
  cout << c << endl;

  return 0;
}
