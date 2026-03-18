/**
 * @file 116A.CPP
 * @brief This program simulates a tram journey to find the maximum passenger capacity required.
 * Algorithm: Iterative simulation of passenger entry and exit, maintaining a current capacity and tracking the maximum observed capacity.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to execute the tram capacity simulation.
 * It reads the number of stops, then iteratively processes passenger changes at each stop
 * to determine the maximum capacity needed.
 */
int main() {
  int n;
  int maxCap = 0;
  int currentCap = 0;
  cin >> n;
  /**
   * @brief Iterates through each tram stop to simulate passenger changes and track capacity.
   * Pre-condition: 'n' represents the total number of stops. 'maxCap' and 'currentCap' are initialized to 0.
   * Invariant: At the start of each iteration, 'currentCap' holds the number of passengers
   * currently in the tram, and 'maxCap' stores the highest capacity recorded so far.
   */
  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;
    currentCap += b;
    currentCap -= a;
    /**
     * @brief Updates 'maxCap' if the 'currentCap' exceeds the previously recorded maximum.
     * Pre-condition: 'currentCap' reflects the passenger count after a stop's changes.
     * Post-condition: 'maxCap' is updated to the greater of its current value or 'currentCap'.
     */
    if (currentCap >= maxCap) {
      maxCap = currentCap;
    }
  }
  cout << maxCap << endl;
  return 0;
}