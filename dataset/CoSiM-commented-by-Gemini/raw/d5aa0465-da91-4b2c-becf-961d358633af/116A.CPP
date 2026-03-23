/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A, "Tram".
 * @details The problem is to calculate the minimum possible capacity of a tram.
 * The input consists of 'n' stops. For each stop, we are given the number of
 * passengers exiting and entering the tram. The program calculates the number
 * of passengers inside the tram after each stop and finds the maximum value,
 * which corresponds to the required capacity.
 *
 * @algorithm Simulation and Tracking Maximum Value
 * @complexity
 * Time: O(N), where N is the number of stops, as the program iterates through each stop once.
 * Space: O(1), as it uses a fixed number of variables.
 */
#include <bits/stdc++.h> // A common non-standard header in competitive programming
using namespace std;

int main() {
  int n; // Number of tram stops
  int a; // Passengers exiting at a stop
  int b; // Passengers entering at a stop
  
  cin >> n;

  int max_capacity = 0; // Tracks the maximum number of passengers ever on the tram
  int current_passengers = 0; // Tracks the current number of passengers

  /**
   * @brief Loop to process each stop.
   * Pre-condition: 'n' is the number of stops to process.
   * Invariant: After each iteration, 'current_passengers' is the number of people on the
   * tram just after it leaves the stop, and 'max_capacity' holds the maximum
   * number of passengers seen so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    // Passengers exit first, then new passengers enter.
    current_passengers -= a;
    current_passengers += b;

    // Update the maximum capacity if the current number of passengers is higher.
    if (current_passengers > max_capacity) {
      max_capacity = current_passengers;
    }
  }

  // Print the minimum required capacity.
  cout << max_capacity << endl;

  return 0; // Indicate successful execution.
}