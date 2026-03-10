/**
 * @file 116A.CPP
 * @brief This program solves the "Tram" problem (Codeforces 116A).
 *   It determines the minimum capacity a tram must have to accommodate all passengers
 *   at any point in time during its journey. Passengers enter and exit at `n` stops.
 *   Problem description: https://codeforces.com/problemset/problem/116/A
 * * Algorithm: Cumulative sum tracking and sorting.
 *   The program first reads all passenger exit/entry data. Then, it simulates the
 *   tram's journey, calculating the passenger count after each stop and storing
 *   these intermediate counts in an array. Finally, it sorts this array to find
 *   the maximum passenger count, which represents the minimum required capacity.
 * Time Complexity: O(N log N) where N is the number of tram stops, dominated by the sorting step.
 * Space Complexity: O(N) for storing the `a`, `b` (exit/entry counts), and `ans` (intermediate passenger counts) arrays.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, such as <iostream>, <algorithm>, <vector>.

using namespace std;

/**
 * @brief Main function to calculate the minimum required tram capacity.
 * Functional Utility: Reads the number of stops and passenger changes at each stop,
 *                     simulates the tram's occupancy, stores intermediate counts,
 *                     then sorts to find and output the peak occupancy (minimum capacity).
 */
int main() {
  int n; // Stores the number of tram stops.
  int i; // Loop counter.
  // `a`, `b`: Arrays to store the number of passengers exiting (`a`) and entering (`b`) at each stop.
  // Max capacity 1000 stops.
  int a[1000], b[1000];
  int x; // `x`: Tracks the current number of passengers inside the tram.
  int t; // `t`: Used to carry over the passenger count from the previous stop.
  int ans[1000]; // `ans`: Array to store the passenger count after each stop.

  // Block Logic: Reads the total number of stops `n`.
  cin >> n;
  /**
   * Block Logic: Reads the number of passengers exiting (`a[i]`) and entering (`b[i]`)
   *              for each of the `n` stops.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: `a` and `b` arrays are populated with all exit/entry data.
   */
  for (i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }

  // Block Logic: Calculates the initial passenger count after the first stop.
  // `x`: current passengers after first stop (passengers entering).
  // `t`: stores this value to be used for the next iteration.
  // `ans[0]`: stores the passenger count after the first stop.
  x = a[0] + b[0]; // Assuming initial passengers are 0, so after first stop it's just `b[0]`.
                   // This might be `b[0]` if `a[0]` is the *exit* from the previous stop which is 0.
  t = x;
  ans[0] = t;
  /**
   * Block Logic: Iterates from the second stop to calculate passenger counts.
   * Pre-condition: `ans[0]` contains the passenger count after the first stop.
   * Invariant: `x` represents the passenger count after the current stop, and `ans[i]` stores this value.
   */
  for (i = 1; i < n; i++) {
    // Functional Utility: Updates the current number of passengers (`x`).
    // It takes the previous passenger count (`t`), subtracts exiting (`a[i]`), and adds entering (`b[i]`).
    x = (t - a[i]) + b[i];
    ans[i] = x; // Stores the updated passenger count.
    t = x;      // Updates `t` to carry over the current passenger count for the next iteration.
  }
  // Functional Utility: Sorts the `ans` array in ascending order.
  // After sorting, the last element `ans[n-1]` will be the maximum passenger count observed.
  sort(ans, ans + n);
  // Block Logic: Prints the maximum value in `ans` (which is the last element after sorting),
  // representing the minimum required capacity.
  cout << ans[n - 1];

  return 0; // Indicates successful program termination.
}