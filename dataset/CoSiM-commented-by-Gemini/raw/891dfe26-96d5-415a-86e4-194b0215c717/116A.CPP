/**
 * @file 116A.CPP
 * @brief Calculates passenger flow on a tram to find its required capacity.
 *
 * @details
 * This program is intended to solve the "Tram" problem, which involves calculating
 * the minimum required capacity of a tram across several stops. At each stop,
 * a number of passengers exit and another number enter. The program simulates
 * this process to find the maximum number of passengers on the tram at any given time.
 *
 * @section
 * Algorithm: Simulation of passenger count stop-by-stop.
 * Time Complexity: O(N), where N is the number of stops, for iterating through the stops.
 * Space Complexity: O(N), as it stores the data for all stops in a vector before processing.
 *
 * @note There appear to be two bugs in this implementation:
 * 1. The variable `maxval` (intended to track peak capacity) is initialized incorrectly.
 * 2. The final output is the number of passengers after the last stop, not the
 *    peak number of passengers, which is the actual required capacity.
 */
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n;
  cin >> n;
  // 'max' tracks the current number of passengers on the tram. It is initialized to 0.
  int max = 0;

  // A vector of pairs to store the exit and entry counts for each of the 'n' stops.
  vector<pair<int, int>> pairs(n);
  for (int i = 0; i < n; i++) {
    cin >> pairs[i].first;  // Number of passengers exiting.
    cin >> pairs[i].second; // Number of passengers entering.
  }

  // 'maxval' is intended to track the peak number of passengers (the required capacity).
  // Its initialization here is likely a bug, as it uses the number of people exiting
  // at the first stop instead of a baseline like 0.
  int maxval = pairs[0].first;

  /**
   * @brief This loop simulates the change in passenger count at each stop.
   *
   * Logic: It updates the current passenger count ('max') and compares it against
   * the highest count seen so far ('maxval').
   */
  for (int i = 0; i < n; i++) {
    // Subtract exiting passengers.
    max -= pairs[i].first;
    // Add entering passengers.
    max += pairs[i].second;
    // If the current passenger count is a new peak, update maxval.
    if (maxval < max)
      maxval = max;
  }

  // The program incorrectly prints the final number of passengers ('max')
  // instead of the peak number of passengers ('maxval').
  cout << max;
  return 0;
}
