/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - Tram.
 *
 * This program calculates the minimum capacity a tram needs to have to avoid
 * overcrowding, given a sequence of passenger exits and entries at each stop.
 * The tram starts empty.
 *
 * Algorithm: Simulation and tracking maximum. The program simulates the passenger
 * flow through the tram. It keeps track of the current number of passengers
 * and updates a maximum value whenever the current passenger count exceeds it.
 * Time Complexity: O(N), where N is the number of tram stops.
 * Space Complexity: O(1), as only a few variables are stored.
 *
 * Author: Andr s Sierra
 * Institution: Instituto Tecnol gico de Santo Domingo (INTEC)
 * Note: This code includes comments indicating its origin from "Codeforces_Andrespengineer".
 */

// Codeforces_Andrespengineer
/* List of solved problems
 *
 * Andr s Sierra
 * Instituto Tecnol gico de Santo Domingo (INTEC)
 *
 */

#include <iostream> // Required for standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace.

int main() {
  /**
   * @brief Main function to execute the Tram problem logic.
   *
   * Reads the number of tram stops 'n'. It then iterates through each stop,
   * reading the number of passengers who exit ('a') and enter ('b').
   * The function calculates the current number of passengers in the tram
   * ('temp') and continuously updates the maximum observed capacity ('capacity').
   * Finally, it prints the maximum passenger count required.
   */
  // Declare variables:
  // 'n' for the number of tram stops (also used as a loop counter).
  // 'capacity' to store the maximum number of passengers observed, initialized to 0.
  // 'a' for passengers exiting at a stop.
  // 'b' for passengers entering at a stop.
  // 'temp' to store the current number of passengers in the tram.
  int n, capacity = 0;
  int a, b, temp;

  // Read the total number of tram stops.
  cin >> n;

  // 'flag' stores the initial value of 'n' to identify the first iteration.
  int flag = n;

  // Block Logic: Process each tram stop.
  // Invariant: The loop iterates 'n' times, processing one stop's data per iteration.
  // 'n' decrements in each step.
  while (n > 0) {
    // Read the number of passengers exiting and entering at the current stop.
    cin >> a >> b;

    // Block Logic: Special handling for the first iteration (first stop).
    // Pre-condition: 'n' equals 'flag' only on the very first iteration.
    if (n == flag) {
      // For the first stop, the initial capacity is simply the number of passengers who enter.
      capacity = b;
      temp = b; // Initialize current passengers with the first entry count.
    } else {
      // For subsequent stops, update the current number of passengers.
      temp = (temp - a) + b;

      // If the current passenger count 'temp' is greater than or equal to
      // the previously recorded maximum 'capacity', update 'capacity'.
      if (temp >= capacity)
        capacity = temp;
    }

    n--; // Decrement the loop counter.
  }

  // Output the maximum passenger count observed, which represents the minimum required capacity.
  cout << capacity << endl;
}
