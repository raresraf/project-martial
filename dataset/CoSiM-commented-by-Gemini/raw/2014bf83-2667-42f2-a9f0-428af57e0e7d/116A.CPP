/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" competitive programming problem.
 *
 * @details This program calculates the minimum required capacity of a tram.
 * It simulates the change in the number of passengers at each stop and finds
 * the maximum number of passengers inside the tram at any point in time. This
 * maximum value represents the tram's minimum required capacity.
 *
 * @section algorithm
 * Algorithm:
 * The solution is a direct simulation of the passenger flow.
 * 1. Read the total number of tram stops, `n`.
 * 2. Initialize two variables: `total` (current passengers) to 0, and `max`
 *    (minimum capacity required) to 0.
 * 3. Loop `n` times, once for each stop.
 * 4. In each iteration, read the number of exiting (`a`) and entering (`b`)
 *    passengers.
 * 5. Update the current number of passengers: `total = total - a + b`.
 * 6. After the update, if the new `total` is greater than `max`, update
 *    `max` to the value of `total`.
 * 7. After the loop finishes, `max` holds the peak number of passengers
 *    observed, which is the minimum capacity needed. Print `max`.
 *
 * @section complexity
 * Time Complexity: O(N), where N is the number of stops, because the program
 * iterates through the stops exactly once.
 * Space Complexity: O(1), as a constant amount of memory is used regardless of
 * the number of stops.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  int n; // Number of tram stops
  int a; // Passengers exiting at a stop
  int b; // Passengers entering at a stop
  int total = 0; // Current number of passengers on the tram
  int max = 0;   // Tracks the maximum number of passengers seen so far

  cin >> n;
  // Block Logic: Simulate the passenger count at each stop.
  // Pre-condition: 'n' is the number of stops to process.
  for (int i = 0; i < n; i++) {
    cin >> a;
    cin >> b;
    // Update the current passenger count after exits and entries.
    total = total - a + b;
    // Invariant: 'max' holds the maximum passenger count found in stops 0 to i.
    if (total > max) {
      max = total;
    }
  }
  // Output the minimum required capacity.
  cout << max;
}
