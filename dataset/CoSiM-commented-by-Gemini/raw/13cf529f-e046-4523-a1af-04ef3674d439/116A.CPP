/**
 * @file 116A.CPP
 * @brief Solution for a problem that calculates the minimum required capacity of a tram.
 * @details This program simulates the passenger flow in a tram over a series of stops.
 * It calculates the maximum number of people inside the tram at any given time to determine
 * the minimum capacity required.
 *
 * Algorithm:
 * 1. Read the number of stops, n.
 * 2. Initialize current capacity `c` and minimum required capacity `m` to 0.
 * 3. Loop n times:
 *    a. Read the number of passengers exiting (a) and entering (b).
 *    b. Update the current number of passengers: c = c - a + b.
 *    c. Update the maximum number of passengers if the current number is greater: m = max(m, c).
 * 4. Print the minimum required capacity, m.
 *
 * Time Complexity: O(n), where n is the number of stops.
 * Space Complexity: O(1), as it uses a constant amount of extra space.
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n;
  cin >> n; // Read the number of tram stops.
  int a, b, c = 0, m = 0; // a: exiting, b: entering, c: current passengers, m: max passengers (capacity)
  
  /**
   * @brief Main loop to process each tram stop.
   * Iterates n times, once for each stop.
   */
  while (n--) {
    cin >> a >> b; // Read passengers exiting and entering at the current stop.
    
    // Update the number of passengers in the tram.
    c = c - a;
    c = c + b;
    
    /**
     * @brief Update the maximum capacity.
     * If the current number of passengers exceeds the previously recorded maximum,
     * update the maximum.
     */
    if (c > m)
      m = c;
  }
  
  // Output the minimum required capacity of the tram.
  cout << m << endl;
  ;
}
