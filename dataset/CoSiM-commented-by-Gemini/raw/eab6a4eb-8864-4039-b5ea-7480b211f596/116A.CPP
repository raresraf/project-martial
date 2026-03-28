/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from Codeforces.
 *
 * This program calculates the minimum possible capacity of a tram to handle
 * passengers over a series of stops. This is equivalent to finding the
 * maximum number of passengers inside the tram at any given time.
 *
 * @algorithm The program simulates the tram's state stop by stop. It maintains
 * a `totalPassenger` count and a `maxPassenger` value. In a loop that runs `n`
 * times (for `n` stops), it reads the number of exiting and entering passengers,
 * updates the `totalPassenger` count, and then updates `maxPassenger` if the
 * new total is greater than the previous maximum.
 *
 * @time_complexity O(N), where N is the number of stops.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, a, b, totalPassenger = 0, maxPassenger = INT_MIN;
  cin >> n;
  /**
   * @block_logic This loop simulates the passenger flow at each of the `n` stops.
   */
  while (n--) {
    cin >> a >> b; // `a` passengers exit, `b` passengers enter.
    // Update the current number of passengers on the tram.
    totalPassenger = (totalPassenger - a) + b;
    // Keep track of the maximum number of passengers seen so far.
    if (totalPassenger > maxPassenger)
      maxPassenger = totalPassenger;
  }
  cout << maxPassenger;
}
