/**
 * @file 116A.CPP
 * @brief Solution to Codeforces Problem 116A: "Tram".
 * Calculates the minimum capacity a tram needs to have, given a series of stops
 * where passengers get on and off. The capacity must be sufficient to hold the
 * maximum number of passengers in the tram at any single point.
 * Algorithm: Simulates the tram's journey, tracking the current number of passengers.
 * The maximum passenger count encountered during the journey determines the minimum capacity.
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1) as only a few variables are used.
 *
 * Problem Link    :   https://codeforces.com/problemset/problem/116/A
 * Contest Link    :   https://codeforces.com/contest/116/problem/A
 * Status          :   Accepted
 * Author          :   Nikhil Nagrale
 * Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
 * Codechef        :   https://www.codechef.com/users/nikhilnagrale2
 * Github          :   https://github.com/nikhilnagrale2
 * Copyright       :   Solve it on your own.
 */

#include <bits/stdc++.h> // Includes various standard libraries for C++, including iostream and algorithm.
using namespace std;

/**
 * @brief Main function to solve the "Tram" problem.
 * Reads the number of stops and for each stop, the number of passengers getting off
 * and on, then calculates and prints the maximum number of passengers at any time.
 */
int main() {
  int n; // Stores the total number of tram stops.
  cin >> n; // Reads the number of stops.
  
  int total = 0; // Stores the current number of passengers in the tram.
  // 'min' is a misnomer; it actually stores the maximum number of passengers observed.
  int min = 0;   // Stores the maximum passenger count observed during the journey.
  
  /**
   * Block Logic: Simulates the tram's journey through each stop.
   * Pre-condition: 'n' is the total number of stops. 'total' and 'min' are initialized to 0.
   * Invariant: After each iteration, 'total' reflects the passenger count after the current stop's
   *            operations, and 'min' holds the highest passenger count seen so far.
   */
  while (n--) { // Loop iterates 'n' times, decrementing 'n' with each iteration.
    int x; // Temporary variable to read passenger counts.
    cin >> x; // Reads the number of passengers getting off ('downs').
    total -= x; // Decrements the current passenger count.
    
    cin >> x; // Reads the number of passengers getting on ('ups').
    total += x; // Increments the current passenger count.
    
    /**
     * Block Logic: Updates the maximum observed passenger count.
     * If the current passenger count ('total') exceeds the previously recorded maximum ('min'),
     * 'min' is updated.
     */
    if (total > min) { // The inline comment "you can use max(a,b) here" suggests using `std::max(min, total)`.
      min = total;     // Update 'min' to the new maximum if current count is higher.
    }
  }
  
  cout << min << endl; // Prints the final maximum number of passengers, which is the minimum required capacity.
  return 0;
}