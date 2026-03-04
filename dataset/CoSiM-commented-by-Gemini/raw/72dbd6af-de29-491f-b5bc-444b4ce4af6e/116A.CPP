/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - "Tram".
 *
 * This program calculates the minimum capacity required for a tram by simulating
 * passenger flow over a series of stops. It keeps a running total of the current
 * passengers and tracks the maximum number of passengers on the tram at any point.
 *
 * @algorithm Simulation with running total and maximum tracking.
 * @time_complexity O(N), where N is the number of stops.
 * @space_complexity O(1).
 */

/*
    Problem Link    :   https://codeforces.com/problemset/problem/116/A
    Contest Link    :   https://codeforces.com/contest/116/problem/A
    Status          :   Accepted

    Author          :   Nikhil Nagrale
    Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
    Codechef        :   https://www.codechef.com/users/nikhilnagrale2
    Github          :   https://github.com/nikhilnagrale2

    Copyright       :   Solve it on your own.
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int total = 0;
  int min = 0; // This variable actually stores the maximum capacity needed.
  /**
   * @brief Simulates the passenger flow for each stop.
   * The loop iterates 'n' times. In each iteration, it reads the number of
   * passengers exiting and entering, updates the current passenger count ('total'),
   * and updates the maximum occupancy ('min') if a new peak is reached.
   */
  while (n--) {
    int x;
    cin >> x;
    total -= x;
    cin >> x;
    total += x;
    if (total > min) // you can use max(a,b) here
      min = total;
  }
  cout << min << endl;
  return 0;
}
