/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem, which calculates the minimum required tram capacity.
 *
 * This program simulates the passenger flow in a tram over a number of stops
 * to determine the maximum number of passengers inside the tram at any given time.
 * This maximum number represents the minimum capacity the tram must have.
 *
 * @algorithm Simulation with in-place array modification.
 * @time_complexity O(N), where N is the number of tram stops, as it iterates through the stops a constant number of times.
 * @space_complexity O(N) to store the passenger data.
 * @note This implementation is functionally correct but unnecessarily complex. It uses multiple loops
 * and overwrites the input array with intermediate passenger counts before finding the maximum value.
 * A simpler approach would be to calculate the current and maximum passenger counts in a single loop.
 */
#include <iostream>
using namespace std;

int main() {
  int arr[1001][2];
  int n;
  cin >> n;
  /**
   * @brief Reads the passenger data for each stop.
   * arr[i][0] = passengers exiting at stop i.
   * arr[i][1] = passengers entering at stop i.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 2; j++)
      cin >> arr[i][j];
  }

  int max = 0; // This variable tracks the current number of passengers.
  /**
   * @brief Simulates passenger flow and overwrites the input array with intermediate passenger counts.
   *
   * This loop calculates the number of passengers after each exit and entry event
   * and stores these running totals back into the `arr` array.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 2; j++) {
      if (j == 0) {
        // Subtract exiting passengers.
        max -= arr[i][j];
      } else {
        // Add entering passengers.
        max += arr[i][j];
      }
      // Overwrite the original exit/entry data with the current passenger count.
      arr[i][j] = max;
    }
  }

  // Initialize maxval with the first calculated passenger count.
  int maxval = arr[0][0];
  /**
   * @brief Finds the maximum value in the modified array.
   * This value represents the peak number of passengers on the tram at any point.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 2; j++) {
      if (maxval < arr[i][j]) {
        maxval = arr[i][j];
      }
    }
  }
  cout << maxval;
  return 0;
}
