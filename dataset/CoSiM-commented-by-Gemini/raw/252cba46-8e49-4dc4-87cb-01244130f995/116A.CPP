/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A, "Tram".
 * @details This program calculates the minimum possible capacity of a tram required
 * to accommodate passengers over a number of stops. It simulates the change in
 * passenger numbers at each stop and finds the maximum number of passengers
 * inside the tram at any one time.
 *
 * @algorithm Iterative simulation to track a running total and find its maximum value.
 * @time_complexity O(n), where 'n' is the number of tram stops, as the program
 * iterates once for each stop.
 * @space_complexity O(1), as a fixed number of variables are used regardless of
 * the number of stops.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {

  int n;
  cin >> n;
  int capacity = 0; // Current number of passengers in the tram.
  int max = 0;      // Maximum number of passengers observed so far.
  int exit, enter;  // Passengers exiting and entering at the current stop.

  /**
   * Block Logic: Simulates the tram's journey across 'n' stops.
   * Invariant: At the start of each iteration, 'capacity' holds the number of
   * passengers arriving at the current stop, and 'max' holds the maximum
   * capacity needed up to the previous stop.
   */
  for (int i = 0; i < n; i++) {
    cin >> exit >> enter;
    
    // Update the current number of passengers.
    capacity = capacity - exit + enter;
    
    // Pre-condition: If the current passenger count exceeds the previous maximum,
    // update the maximum required capacity.
    if (capacity > max)
      max = capacity;
  }
  
  // Output the minimum required capacity for the tram.
  cout << max << endl;
  
  return 0;
}