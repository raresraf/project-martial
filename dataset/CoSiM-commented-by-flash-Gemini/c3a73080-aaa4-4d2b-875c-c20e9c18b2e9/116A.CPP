/**
 * @file 116A.CPP
 * @brief This program solves the "Tram" problem (Codeforces 116A).
 *   It determines the minimum capacity a tram must have to accommodate all passengers
 *   at any point in time during its journey. Passengers enter and exit at `n` stops.
 *   Problem description: https://codeforces.com/problemset/problem/116/A
 * * Algorithm: Cumulative sum tracking and sorting.
 *   The program simulates the tram's journey, maintaining a running total of passengers
 *   currently inside the tram. It stores all intermediate passenger counts in a vector,
 *   then sorts this vector to find the maximum value, which represents the minimum
 *   required capacity.
 * Time Complexity: O(N log N) where N is the number of tram stops, due to sorting the vector
 *   of intermediate passenger counts.
 * Space Complexity: O(N) for storing the vector `d` of intermediate passenger counts.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, such as <iostream>, <vector>, <algorithm>.
using namespace std;

/**
 * @brief Main function to calculate the minimum required tram capacity.
 * Functional Utility: Reads the number of stops and passenger changes at each stop,
 *                     simulates the tram's occupancy, stores intermediate counts,
 *                     then sorts to find and output the peak occupancy (minimum capacity).
 */
int main() {
  int a, b, c, i, j, l, k, o = 0; // `a`: total stops (renamed from `n`), `b`: exiting, `c`: entering.
                                // `i, j, l, k, o`: declared but unused loop counters/variables.
  int e = 0; // `e`: Tracks the current number of passengers inside the tram.
  // Block Logic: Reads the total number of stops (`a`).
  cin >> a;
  vector<int> d; // `d`: Stores intermediate passenger counts after each stop.
  /**
   * Block Logic: Iterates `a` times to simulate each tram stop.
   * Pre-condition: `a` is a non-negative integer representing the total stops.
   * Invariant: `e` updates with current passengers, and `d` stores this `e` value.
   * Inline: `a--` decrements the counter and evaluates to true as long as `a` is greater than 0.
   */
  while (a--) {
    // Block Logic: Reads the number of passengers exiting (`b`) and entering (`c`) at the current stop.
    cin >> b;
    cin >> c;
    // Functional Utility: Updates the current number of passengers (`e`). Passengers exiting are subtracted, entering are added.
    e = e - b + c;
    d.push_back(e); // Stores the current passenger count in the vector.
  }
  // Functional Utility: Sorts the vector `d` in ascending order.
  // After sorting, the last element will be the maximum passenger count observed.
  sort(d.begin(), d.end());
  // Block Logic: Prints the maximum value in `d` (which is the last element after sorting),
  // representing the minimum required capacity.
  cout << d[d.size() - 1];
}