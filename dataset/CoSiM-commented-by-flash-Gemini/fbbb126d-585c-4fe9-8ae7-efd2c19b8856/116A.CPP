/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem from Codeforces (Problem 116A).
 * Problem Statement: Given the number of passengers leaving and entering the tram at each stop,
 *                    determine the minimum capacity the tram should have.
 * Algorithm: Simulation. Iterates through each stop, updating the current number of passengers
 *            and tracking the maximum number of passengers observed at any point.
 * Time Complexity: O(N) where N is the number of stops.
 * Space Complexity: O(1) for constant extra variables (excluding the array `a` which is unused for the solution logic).
 * Domain: Competitive Programming, Simulation.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to solve the Tram problem.
 * Functional Utility: Calculates the maximum number of passengers on the tram at any given time
 *                     to determine the minimum required capacity.
 */
int main() {
  // Variables:
  // n: Number of tram stops.
  // i: Loop counter.
  // x: Number of passengers leaving at the current stop.
  // y: Number of passengers entering at the current stop.
  // entery: Current number of passengers on the tram.
  // leaving: Temporary variable to hold passengers after leaving (not strictly needed due to direct calculation).
  // m: Maximum number of passengers observed on the tram so far.
  int n, i, x, y, current_passengers = 0; // Renamed 'entery' to 'current_passengers' for clarity
  int max_passengers = 0;                 // Renamed 'm' to 'max_passengers' for clarity
  cin >> n;

  // Block Logic: Iterates through each tram stop.
  // Invariant: `current_passengers` holds the number of people on the tram before passengers
  //            leave and enter at the current stop. `max_passengers` holds the maximum
  //            passengers seen up to the previous stop.
  for (i = 0; i < n; i++) {
    cin >> x >> y; // Read passengers leaving (x) and entering (y).
    
    // Functional Utility: Update the number of passengers on the tram.
    // Logic: Passengers leaving are subtracted, and passengers entering are added.
    current_passengers = current_passengers - x + y;

    // Block Logic: Update the maximum passengers if the current count is higher.
    // Invariant: `max_passengers` always stores the highest number of passengers
    //            observed on the tram up to the current stop.
    if (current_passengers > max_passengers) {
      max_passengers = current_passengers;
    }
  }
  // Block Logic: Output the maximum number of passengers observed.
  cout << max_passengers << endl;

  return 0;
}
