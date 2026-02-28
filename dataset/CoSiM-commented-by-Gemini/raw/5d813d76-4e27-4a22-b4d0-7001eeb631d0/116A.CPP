/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" problem from Codeforces.
 * @details This program calculates the minimum required capacity of a tram. It does this
 * by simulating the change in passenger numbers at each stop and keeping track of the
 * maximum number of passengers inside the tram at any point in time.
 *
 * @algorithm Iterative Simulation
 * 1. Read the number of stops, `n`.
 * 2. Initialize `current_passengers` to 0 and `max_capacity` to 0.
 * 3. Loop `n` times, once for each stop.
 * 4. In each iteration, read the number of exiting (`a`) and entering (`b`) passengers.
 * 5. Update the current number of passengers: `current_passengers = current_passengers - a + b`.
 * 6. After the update, if `current_passengers` is greater than `max_capacity`, update
 *    `max_capacity` to the new `current_passengers` value.
 * 7. After the loop, `max_capacity` will hold the minimum capacity required for the tram.
 *
 * @complexity
 * - Time: O(N), where N is the number of stops, because of the single loop.
 * - Space: O(1), as it only uses a few variables to store the state.
 */

#include <iostream>
using namespace std;

/**
 * @brief The main function.
 * @note The use of `main()` without an `int` return type is non-standard in C++.
 * The standard declaration is `int main()`.
 */
main() {
  // n: the number of tram stops.
  int n;
  cin >> n;
  
  // max_passengers: tracks the peak number of passengers seen so far (the required capacity).
  // current_passengers: tracks the number of people on the tram at any given time.
  int max_passengers = 0, current_passengers = 0;
  
  /**
   * @brief Simulate the tram's journey stop by stop.
   */
  // Invariant: `current_passengers` holds the count after the (n - (initial_n - 1)) stop.
  // `max_passengers` holds the peak count observed up to that point.
  while (n > 0) {
    // a: passengers exiting, b: passengers entering.
    int a, b;
    cin >> a >> b;
    
    // Update the number of passengers currently on the tram.
    current_passengers = current_passengers - a + b;
    
    // Pre-condition: Check if the current passenger count is a new maximum.
    if (max_passengers < current_passengers) {
      max_passengers = current_passengers;
    }
    
    n -= 1;
  }
  
  // Print the maximum number of passengers observed, which is the minimum required capacity.
  cout << max_passengers << endl;
}
