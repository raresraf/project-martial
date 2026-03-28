/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 *
 * This program determines the minimum capacity a tram needs to have to accommodate
 * passengers over a series of stops. It reads the number of passengers exiting
 * and entering at each stop and finds the maximum number of people inside the
 * tram at any given time.
 *
 * @b Algorithm: The program iteratively calculates the net change in passengers
 *    at each stop. It maintains a running total of the number of passengers on
 *    the tram and keeps track of the highest number encountered. This is a
 *    simulation of passenger flow.
 * @b Time_Complexity: O(N), where N is the number of tram stops, because it
 *    iterates through the stops a constant number of times.
 * @b Space_Complexity: O(N) to store the entry and exit counts for each stop.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, a[100000], b[100000], c[100000], max;
  cin >> n;
  for (i = 0; i < n; i++)
    cin >> a[i] >> b[i];
  c[-1] = 0;
  /**
   * @brief Loop to calculate the number of passengers at each stop.
   * `c[i]` stores the total number of people on the tram after the i-th stop.
   * `a[i]` is the number of people getting off, and `b[i]` is the number of people getting on.
   */
  for (i = 0; i < n; i++) {
    c[i] = b[i] - a[i] + c[i - 1];
  }
  max = c[0];
  /**
   * @brief Loop to find the maximum number of passengers.
   * This loop iterates through the passenger counts at each stop to find the peak.
   */
  for (i = 0; i < n; i++) {
    if (c[i] > max) {
      max = c[i];
    }
  }
  cout << max;
  return 0;
}
