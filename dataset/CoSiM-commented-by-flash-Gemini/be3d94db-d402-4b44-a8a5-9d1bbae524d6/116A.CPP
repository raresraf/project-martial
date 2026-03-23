/**
 * @file 116A.CPP
 * @brief This competitive programming solution calculates the minimum capacity needed for a tram
 *        to accommodate all passengers given a sequence of entries and exits at various stops.
 *        This version uses global arrays for passenger data.
 * Algorithm: Simulation. The code simulates the number of passengers currently in the tram
 *            at each stop and keeps track of the maximum number of passengers observed
 *            at any single point in time.
 * Time Complexity: O(N), where N is the number of stops, as the program iterates through
 *                  each stop once to update passenger counts.
 * Space Complexity: O(N), for storing the `a` (exits) and `b` (entries) arrays.
 */
#include <algorithm> // Required for std::max (though not explicitly used in this simplified version, typical for such problems).
#include <iostream>  // Required for standard input/output operations (cin, cout).

using namespace std;

#define ll long long; // Macro for long long integer type, though not explicitly used for variables here.

// Global variables to store the number of stops and passenger changes at each stop.
// The problem statement usually implies N <= 1000, so arrays of this size are common.
int n;         // Total number of stops.
int a[1000];   // Array to store passengers exiting at each stop.
int b[1000];   // Array to store passengers entering at each stop.

/**
 * @brief Calculates the maximum number of passengers present in the tram at any point.
 * Functional Utility: Encapsulates the core logic for simulating passenger flow and
 *                     determining the peak capacity requirement.
 */
void solve() {
  int max = 0;             // Stores the maximum number of passengers observed so far.
  int currentPassengers = 0; // Stores the current number of passengers in the tram.
  // Block Logic: Iterates through each stop to simulate passenger movement and track maximum.
  // Invariant: `currentPassengers` always reflects the current count, and `max` stores the
  //            highest `currentPassengers` value encountered up to the current stop.
  for (int i = 0; i < n; i++) {
    currentPassengers += b[i] - a[i]; // Updates the current passenger count (entries - exits).
    // Block Logic: Updates `max` if `currentPassengers` exceeds the current maximum.
    if (max < currentPassengers)
      max = currentPassengers;
  }
  cout << max; // Prints the overall maximum capacity required.
}

/**
 * @brief Main function of the program. Handles input reading and calls the solver function.
 * Functional Utility: Serves as the entry point for execution, preparing data for `solve()`.
 */
int main() {
  cin >> n; // Reads the number of stops.
  // Block Logic: Reads the passenger exit ('a') and entry ('b') counts for each stop.
  // Invariant: After this loop, arrays `a` and `b` are populated with input data.
  for (int i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }
  solve();    // Calls the function to solve the problem.
  return 0; // Indicates successful program termination.
}
