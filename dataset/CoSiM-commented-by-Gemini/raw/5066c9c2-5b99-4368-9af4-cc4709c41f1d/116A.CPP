/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from Codeforces.
 * @details The problem requires finding the minimum possible capacity of a tram
 * by calculating the maximum number of passengers inside the tram at any point
 * in its route. The tram makes n stops.
 *
 * @algorithm
 * 1. Read the number of stops, n.
 * 2. Initialize a variable `c` to 0 to track the current number of passengers.
 * 3. Initialize a variable `m` to 0 to track the maximum number of passengers
 *    seen so far. This will be the required capacity.
 * 4. Loop n times for each stop:
 *    a. Read the number of passengers exiting (`a`) and entering (`b`).
 *    b. Update the current passenger count: `c = c - a + b`.
 *    c. Compare the new current count `c` with the maximum `m` and update `m`
 *       if `c` is larger.
 * 5. After the loop, `m` will hold the maximum number of passengers that were
 *    on the tram at any one time. Print `m`.
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of stops, because we process
 * each stop once.
 * Space Complexity: O(1), as a constant number of variables are used.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n; // Number of tram stops.
  int a, b; // 'a' passengers exit, 'b' passengers enter.
  int c(0); // Current number of passengers on the tram.
  int m(0); // Maximum number of passengers observed so far (the required capacity).
  
  cin >> n;

  /**
   * @block
   * @brief Simulate the passenger flow at each stop.
   * @invariant At the beginning of each iteration, `c` holds the number of
   * passengers arriving at the stop, and `m` holds the max capacity needed
   * for all previous stops.
   */
  while (n--) {
    cin >> a >> b;
    // Update the current passenger count.
    c = c - a; // Passengers exit.
    c = c + b; // Passengers enter.
    
    // Update the maximum capacity if the current count is higher.
    if (c > m) {
      m = c;
    }
  }

  // Print the minimum required capacity.
  cout << m;
  return 0;
}
