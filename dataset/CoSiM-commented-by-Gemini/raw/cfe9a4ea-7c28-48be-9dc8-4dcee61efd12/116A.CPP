/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 *
 * This program solves the "Tram" problem. It simulates the number of passengers
 * inside a tram over several stops. At each stop, some passengers exit and others
 * enter. The goal is to find the maximum number of passengers on the tram at any
 * given time, which corresponds to the minimum capacity it must have.
 *
 * @b Algorithm: Simulation
 *   The code iterates through each stop, updating the current passenger count
 *   and tracking the maximum count observed.
 * @b Time_Complexity: O(N), where N is the number of stops.
 * @b Space_Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: the number of tram stops.
  int n;
  cin >> n;
  
  // a: passengers exiting, b: passengers entering.
  int a, b;
  // maxn: tracks the maximum number of passengers seen so far.
  int maxn = 0;
  // total: the current number of passengers on the tram.
  int total = 0;
  
  /**
   * @brief Loop to simulate the passenger flow at each stop.
   * @note The loop condition `i <= n` runs n+1 times, which may be a bug
   * if the problem specifies exactly 'n' stops. It should likely be `i < n`.
   */
  for (int i = 0; i <= n; i++) {
    cin >> a >> b;
    // Update the current passenger count.
    total -= a;
    total += b;
    // Update the maximum capacity required.
    maxn = max(total, maxn);
  }
  
  // Output the minimum required capacity.
  cout << maxn << endl;
  return 0;
}
