/**
 * @file 116A.cpp
 * @brief Solution to the "Tram" problem (Codeforces 116A).
 * This program calculates the maximum capacity reached inside a tram given a sequence of passenger entries and exits at various stops.
 * Algorithm: Iterative simulation.
 * Time Complexity: O(N), where N is the number of tram stops.
 * Space Complexity: O(1), as only a few variables are used to store the current passenger count and maximum capacity.
 */
// 116A.cpp
// A. Tram
// Author: Ikki
// May 31, 2015

#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n;
  // Functional Utility: Reads the total number of tram stops.
  cin >> n;
  // Initialize current passenger count and maximum capacity observed.
  int val = 0, ans = 0;
  /**
   * Block Logic: Simulates the passenger flow through each tram stop.
   * Pre-condition: `n` represents the total number of stops.
   * Invariant: `val` stores the current number of passengers, `ans` stores the maximum `val` encountered so far.
   */
  for (int i = 0; i < n; i++) {
    int a, b;
    // Functional Utility: Reads the number of passengers exiting (`a`) and entering (`b`) at the current stop.
    cin >> a >> b;
    // Update the current number of passengers in the tram.
    val -= a; // Passengers exiting the tram.
    val += b; // Passengers entering the tram.
    // Update the maximum capacity observed so far.
    ans = max(ans, val);
  }
  // Output the final maximum capacity reached.
  cout << ans << endl;
  return 0;
}