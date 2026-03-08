/**
 * @491d80b6-0551-4964-9555-cdd372eaeb6f/116A.CPP
 * @brief This program calculates the minimum capacity required for a tram
 *        given a sequence of passengers entering and exiting.
 * Algorithm: Iterative simulation of passenger flow, tracking current occupancy and maximum observed occupancy.
 * Time Complexity: O(N) where N is the number of stops, as it involves a single pass through the input.
 * Space Complexity: O(1) for storing a few integer variables.
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  // Declare variables:
  // n: Total number of tram stops.
  // enter: Number of passengers entering at a stop.
  // exit: Number of passengers exiting at a stop.
  // result: Current number of passengers in the tram.
  // capasity: Maximum number of passengers observed in the tram (the minimum capacity required).
  int n, enter, exit, result = 0, capasity = 0;

  // Functional Utility: Read the total number of stops.
  cin >> n;

  // Block Logic: Iterate through each tram stop.
  // Invariant: 'result' holds the current number of passengers, 'capasity' holds the maximum observed.
  while (n--) { // Loop 'n' times for each stop.
    // Functional Utility: Read the number of passengers exiting and entering at the current stop.
    cin >> exit >> enter;

    // Action: Update the current number of passengers in the tram.
    result = result + (enter - exit);

    // Pre-condition: Check if the current number of passengers exceeds the previously recorded maximum capacity.
    if (result > capasity) {
      capasity = result; // Update the maximum capacity if current occupancy is higher.
    }
  }

  // Functional Utility: Output the minimum capacity required for the tram.
  cout << capasity;

  return 0; // Indicate successful program termination.
}
