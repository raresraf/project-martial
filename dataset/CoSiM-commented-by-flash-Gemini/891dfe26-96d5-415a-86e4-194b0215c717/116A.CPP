/**
 * @file 116A.CPP
 * @brief Calculates passenger count on a tram line to determine capacity.
 *
 * This program simulates the number of passengers on a tram over a series of stops.
 * For each stop, it's given the number of passengers exiting and entering.
 * The goal is to find the maximum number of passengers on the tram at any point,
 * which corresponds to the minimum required capacity.
 *
 * @algorithm Single-pass simulation.
 * @time_complexity O(N), where N is the number of tram stops, as it processes each stop once.
 * @space_complexity O(N) to store the exit and enter data for all stops.
 */
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n;
  cin >> n;
  // @brief Tracks the current number of passengers on the tram. Initialized to 0.
  int max = 0;

  // @brief A vector of pairs to store the number of passengers exiting (first) and entering (second) at each stop.
  vector<pair<int, int>> pairs(n);
  /**
   * @brief Block Logic: Reads the passenger data for all stops.
   * @invariant `pairs` vector is populated with n pairs of (exit, enter) counts.
   */
  for (int i = 0; i < n; i++) {
    cin >> pairs[i].first;
    cin >> pairs[i].second;
  }

  // @brief Tracks the peak number of passengers seen so far.
  // Note: Incorrectly initialized with the number of passengers exiting at the first stop.
  int maxval = pairs[0].first;

  /**
   * @brief Block Logic: Simulates the passenger flow through all the stops.
   *
   * This loop updates the current passenger count (`max`) at each stop and
   * tracks the maximum value seen (`maxval`).
   * @invariant `max` holds the current passenger count after stop `i`, and `maxval`
   * holds the maximum count seen up to stop `i`.
   */
  for (int i = 0; i < n; i++) {
    max -= pairs[i].first;  // Passengers exit.
    max += pairs[i].second; // Passengers enter.

    // Block Logic: Updates the peak passenger count if the current count is higher.
    if (maxval < max)
      maxval = max;
  }
  // Functional Utility: Prints the final number of passengers on the tram after the last stop.
  // Note: This is likely a bug. To find the minimum capacity, this should print `maxval`.
  cout << max;
  return 0;
}
