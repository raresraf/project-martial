/**
 * @file 116A.CPP
 * @brief An optimal solution to the "Tram" problem from Codeforces.
 *
 * This program calculates the minimum required capacity of a tram by simulating
 * the passenger flow over a series of stops. It determines the maximum number
 * of passengers inside the tram at any single point in time.
 *
 * @algorithm Single-pass simulation
 * @time_complexity O(N), where N is the number of tram stops, as it uses a single loop.
 * @space_complexity O(1), as it only uses a few variables to track state, regardless of the number of stops.
 */
#include <iostream>

using namespace std;

int main() {
  int n, a, b;
  int passengers = 0; // Tracks the current number of passengers on the tram.
  int capacity = 0;   // Tracks the maximum number of passengers seen so far (the required capacity).
  cin >> n;

  /**
   * @brief Simulates the passenger flow through all stops in a single pass.
   *
   * For each stop, it reads the number of exiting (a) and entering (b) passengers,
   * updates the current passenger count, and adjusts the required capacity if the
   * current count exceeds the previous maximum.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    passengers -= a; // Passengers exit.
    passengers += b; // Passengers enter.
    
    // If the current number of passengers is a new maximum, update the capacity.
    if (passengers > capacity)
      capacity = passengers;
  }

  cout << capacity;

  return 0;
}
