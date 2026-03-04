/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - "Tram".
 *
 * This program calculates the minimum possible capacity of a tram. It simulates
 * the passenger flow over a number of stops. At each stop, some passengers exit
 * and some enter. The program tracks the current number of passengers and finds
 * the maximum number of passengers on the tram at any given time, which
 * represents the minimum required capacity.
 *
 * @algorithm Simulation with running total and maximum tracking.
 * @time_complexity O(N), where N is the number of stops, because it processes
 *                  each stop once.
 * @space_complexity O(1), as it only uses a few variables to store the state.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to calculate the minimum tram capacity.
 */
int main() {
  int stops;
  // Read the total number of stops.
  cin >> stops;
  int maximum = -1;
  int cur = 0;
  /**
   * @brief Simulates the passenger flow at each stop.
   * This loop iterates for each stop. In each iteration, it reads the number
   * of passengers exiting and entering, updates the current passenger count,
   * and then updates the maximum passenger count if the current count is higher.
   */
  while (stops--) {
    int ups, downs;
    cin >> downs >> ups;
    cur = cur + ups - downs;
    maximum = max(maximum, cur);
  }
  // Print the calculated minimum capacity.
  cout << maximum;
  return 0;
}
