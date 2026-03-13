/**
 * @file 116A.CPP
 * @brief Solution for CodeForces Problem 116A - "Tram".
 * 
 * @details
 * This program calculates the minimum required capacity of a tram.
 * It simulates the number of passengers on a tram over several stops. At each stop,
 * a number of passengers exit and a number of passengers enter. The program tracks the
 * current number of passengers and finds the maximum value seen during the trip.
 *
 * @algorithm
 * 1. Read the number of stops, `x`.
 * 2. Initialize `temp1` (current passengers) and `temp2` (maximum passengers so far) to 0.
 * 3. Loop `x` times for each stop:
 *    a. Read the number of passengers exiting (`n1`) and entering (`n2`).
 *    b. Update the current number of passengers: `temp1 = temp1 - n1 + n2`.
 *    c. If the current number of passengers `temp1` is greater than the maximum `temp2`,
 *       update `temp2` to `temp1`.
 * 4. Print the final maximum `temp2`.
 *
 * @time_complexity O(N), where N is the number of stops.
 * @space_complexity O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int x;      // Number of tram stops.
  int n1;     // Passengers exiting at a stop.
  int n2;     // Passengers entering at a stop.
  int temp1 = 0; // Tracks the current number of passengers on the tram.
  int temp2 = 0; // Tracks the maximum number of passengers seen so far.
  
  cin >> x;

  /**
   * @brief Main loop to simulate the tram journey stop by stop.
   * Invariant: At the start of each iteration, `temp1` is the passenger count from the
   * previous stop, and `temp2` is the maximum count seen up to that point.
   */
  while (x > 0) {
    cin >> n1 >> n2;
    // Update the current number of passengers.
    temp1 = temp1 - n1 + n2;
    
    // If the current passenger count exceeds the max, update the max.
    if (temp1 > temp2) {
      temp2 = temp1;
    }
    x--;
  }
  
  // Output the minimum required capacity, which is the maximum passengers at any point.
  cout << temp2 << endl;
  return 0;
}
