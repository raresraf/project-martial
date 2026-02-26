/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem.
 * @details This program calculates the minimum required capacity of a tram based on
 * the number of passengers entering and exiting at each stop. It simulates the
 * passenger count over a series of stops and finds the maximum number of
 * people on the tram at any given time.
 *
 * @algorithm Iterative simulation of passenger count.
 * @complexity
 * Time Complexity: O(N), where N is the number of tram stops, as the program
 * iterates through each stop once.
 * Space Complexity: O(1), as the program uses a fixed number of variables
 * regardless of the input size.
 */

// A non-standard header common in competitive programming that includes most
// standard library headers. For production code, it's better to include
// specific headers like <iostream>.
#include <bits/stdc++.h>

// Macros for type and function shortcuts. Not used in this specific file but
// are likely part of a standard template for the user.
#define ll long long int
#define pb push_back

// Brings all names from the std namespace into the current scope.
// Common in competitive programming for brevity, but can cause name clashes
// in larger projects.
using namespace std;

int main() {
  // n: the total number of tram stops.
  // a: the number of passengers exiting at the current stop.
  // b: the number of passengers entering at the current stop.
  // current_passengers: tracks the number of people currently on the tram.
  // min_capacity: stores the maximum number of passengers observed so far.
  int n, a, b, current_passengers = 0, min_capacity = -1;

  // Read the total number of stops.
  cin >> n;

  /**
   * @block
   * @brief Main simulation loop.
   * @invariant Iterates through each of the `n` stops to update passenger count.
   */
  for (int i = 0; i < n; i++) {
    // Read the number of passengers exiting and entering at the current stop.
    cin >> a >> b;

    // Update the current number of passengers on the tram.
    current_passengers = current_passengers - a;
    current_passengers = current_passengers + b;

    /**
     * @block
     * @brief Update the required capacity.
     * @pre `current_passengers` holds the number of people on the tram after
     *      the exchange at the current stop.
     * @post `min_capacity` is updated if the current passenger count is the
     *       highest seen so far.
     */
    if (current_passengers > min_capacity)
      min_capacity = current_passengers;
  }

  // Output the maximum number of passengers, which is the minimum required capacity.
  cout << min_capacity << endl;
}