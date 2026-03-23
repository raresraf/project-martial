/**
 * @file 116A.CPP
 * @brief This competitive programming solution calculates the minimum capacity needed for a tram
 *        to accommodate all passengers given a sequence of entries and exits at various stops.
 * Algorithm: Simulation. The code simulates the number of passengers currently in the tram
 *            at each stop and keeps track of the maximum number of passengers observed
 *            at any single point in time.
 * Time Complexity: O(N), where N is the number of stops, as the program iterates through
 *                  each stop once to update passenger counts.
 * Space Complexity: O(1), as only a few fixed-size integer variables are used to store
 *                   current and maximum passenger counts.
 */
#include <algorithm> // Required for std::max function.
#include <iostream>  // Required for standard input/output operations (cin, cout).
using namespace std;

/**
 * @brief Solves the tram capacity problem for a single test case.
 * Functional Utility: Encapsulates the core logic to calculate the maximum number of passengers
 *                     present in the tram at any given time.
 */
void solved() {
  int a, b; // 'a': passengers exiting at a stop; 'b': passengers entering at a stop.
  int n;    // Total number of stops.
  cin >> n; // Reads the number of stops.
  int temp = 0; // Current number of passengers in the tram.
  int MAX = 0;  // Maximum number of passengers observed in the tram so far.
  // Block Logic: Iterates through each stop to simulate passenger movement and track maximum.
  // Invariant: `temp` always holds the current passenger count, and `MAX` holds the
  //            highest `temp` value encountered up to the current stop.
  for (int i = 0; i < n; i++) {
    cin >> a >> b;         // Reads passengers exiting and entering at the current stop.
    temp = temp + b - a;   // Updates the current passenger count.
    MAX = max(temp, MAX);  // Updates the maximum observed passenger count.
  }
  cout << MAX; // Prints the overall maximum capacity required.
}

/**
 * @brief Main function of the program.
 * Functional Utility: Entry point for execution, calls the `solved` function.
 */
int main() {
  solved();   // Calls the function to solve the problem.
  return 0; // Indicates successful program termination.
}
