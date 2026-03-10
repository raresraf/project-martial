/**
 * @file 116A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Tram").
 * This program calculates the minimum possible capacity of a tram by tracking
 * the number of passengers at each stop.
 *
 * @algorithm Simulation with Maximum Tracking. The program simulates the change
 * in passenger count at each stop and keeps track of the maximum number of
 * passengers on the tram at any given time.
 *
 * @time_complexity O(N), where N is the number of tram stops, due to a single
 * loop processing each stop.
 * @space_complexity O(1), as it only uses a few integer variables to store state.
 */
#include <iostream>
using namespace std;

int main() {
  int n; // Number of tram stops
  int currentPass = 0; // Number of passengers currently on the tram
  int maxCap = 0;      // Minimum capacity required (maximum passengers at any point)
  cin >> n;

  /**
   * @block
   * @brief Simulates the passenger flow at each stop.
   *
   * Invariant: `currentPass` holds the number of passengers on the tram just
   * after the events of the current stop. `maxCap` holds the peak number of
   * passengers seen up to the previous stop.
   */
  while (n--) {
    int a, b; // a: exiting passengers, b: entering passengers
    cin >> a >> b;

    // Update the current number of passengers.
    currentPass += b - a;

    // Update the maximum capacity if the current passenger count is a new peak.
    maxCap = max(maxCap, currentPass);
  }

  cout << maxCap << endl;
  return 0;
}
