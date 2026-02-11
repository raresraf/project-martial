/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 * @details This program solves the "Tram" problem from Codeforces. It simulates
 * the passenger count over a series of stops to find the maximum number of
 * passengers on board at any given time, which corresponds to the minimum
 * required capacity.
 *
 * @see https://codeforces.com/problemset/problem/116/A
 *
 * Algorithm:
 * 1. Read the total number of stops.
 * 2. Initialize a variable `pas` to track the current number of passengers and
 *    a variable `maks` to track the maximum number of passengers seen so far.
 * 3. Loop for each stop:
 *    a. Read the number of passengers exiting (`a`) and entering (`b`).
 *    b. Update the current passenger count by subtracting `a` and adding `b`.
 *    c. If the new passenger count `pas` is greater than the recorded max `maks`,
 *       update `maks`.
 * 4. After the loop, `maks` holds the minimum required capacity. Print its value.
 *
 * Time Complexity: O(N), where N is the number of stops, due to a single loop.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: the number of stops.
  int n;
  // pas: the current number of passengers on the tram.
  int pas = 0;
  // maks: the maximum number of passengers seen, representing the required capacity.
  int maks = 0;
  
  cin >> n;
  
  /**
   * @brief Main loop to simulate passenger flow at each stop.
   *
   * Pre-condition: 'n' is the number of stops to simulate.
   * Invariant: After each iteration, 'pas' is the current passenger count,
   *            and 'maks' is the maximum count observed up to that point.
   */
  while (n--) {
    // a: passengers exiting, b: passengers entering.
    int a, b;
    cin >> a >> b;
    
    // Update the current passenger count.
    pas -= a;
    pas += b;
    
    // If the current count is a new maximum, record it.
    if (pas > maks)
      maks = pas;
  }
  
  // Print the result.
  cout << maks << "\n";
}
