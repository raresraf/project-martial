/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A - Tram.
 * @description This program calculates the minimum required capacity of a tram by tracking the number of passengers at each stop. It determines the maximum number of passengers simultaneously on the tram, which corresponds to the minimum capacity needed.
 * Algorithm: The code iterates through each tram stop, updating a counter for the current number of passengers (`space`). After each stop, it compares the current passenger count to a running maximum (`maxSpace`) and updates it if necessary.
 * Time Complexity: O(N), where N is the number of tram stops, as it requires a single pass.
 * Space Complexity: O(1), using a fixed number of variables regardless of input size.
 */
#include <iostream>

using std::cin;
using std::cout;
using std::endl;
int main() {
  // n: the total number of stops.
  // out: number of passengers exiting.
  // in: number of passengers entering.
  int n, out, in;
  // space: tracks the current number of passengers inside the tram.
  int space = 0;
  cin >> n;
  // maxSpace: stores the maximum number of passengers observed at any point, representing the required minimum capacity.
  int maxSpace = 0;
  /**
   * Block Logic: Loop through each stop to simulate passenger flow.
   * Invariant: At the beginning of each iteration, 'space' holds the passenger count from the previous stop,
   * and 'maxSpace' holds the maximum count observed so far.
   */
  for (int i = 0; i != n; ++i) {
    // Read the number of passengers exiting and entering at the current stop.
    cin >> out >> in;
    // Update the current number of passengers.
    space -= out;
    space += in;
    // If the current passenger count exceeds the maximum recorded capacity, update the maximum.
    if (maxSpace < space) {
      maxSpace = space;
    }
  }
  // Output the final calculated minimum capacity.
  cout << maxSpace;
  return 0;
}
