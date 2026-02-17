/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem (Codeforces 116A) by calculating the minimum tram capacity required.
 * @details This program simulates the number of passengers on a tram over a series of stops. It reads the number of passengers exiting and entering at each stop and determines the maximum number of passengers on the tram at any given time, which corresponds to the minimum required capacity.
 *
 * @algorithm Iterative Simulation
 * @description The program iterates through each tram stop, updating the current number of passengers and tracking the maximum number of passengers encountered.
 * @complexity
 * - Time: O(N), where N is the number of stops, because it processes each stop once.
 * - Space: O(1), as it uses a fixed number of variables to store state.
 */
#include <iostream>
using namespace std;

int main() {
  int n, in, out, ans = 0, max = 0;
  cin >> n;

  /**
   * @brief This loop simulates the passenger flow at each of the 'n' tram stops.
   * @invariant At the beginning of each iteration `i`, `ans` holds the number of passengers on the tram as it arrives at stop `i`, and `max` holds the minimum capacity required for the first `i-1` stops.
   */
  for (int i = 0; i < n; i++) {
    cin >> out >> in;
    // Update the current number of passengers on the tram.
    ans = ans - out + in;
    // Update the maximum number of passengers seen so far. This represents the minimum required capacity.
    max = ans > max ? ans : max;
  }

  // Output the minimum required tram capacity.
  cout << max << endl;

  return 0;
}
