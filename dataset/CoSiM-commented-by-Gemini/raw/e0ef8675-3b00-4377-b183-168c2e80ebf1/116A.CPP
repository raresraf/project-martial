/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A - Tram.
 * @description This program calculates the minimum required capacity of a tram based on the number of passengers entering and exiting at each stop.
 * The problem is to find the maximum number of people inside the tram at any given time.
 * Algorithm: The solution iterates through each stop, maintaining a running count of the passengers currently on the tram. It updates a capacity variable with the maximum passenger count observed after each stop.
 * Time Complexity: O(N), where N is the number of tram stops.
 * Space Complexity: O(1), as the program uses a constant amount of extra space regardless of the input size.
 */
#include <iostream>
#include <algorithm>
using namespace std;

int main(int argc, char const *argv[]) {
  // n: the number of tram stops.
  int n;
  cin >> n;
  // minCap: stores the maximum number of passengers on the tram at any point, which determines the minimum required capacity.
  int minCap = 0;
  // a: passengers exiting, b: passengers entering.
  int a, b;
  // sum: the current number of passengers on the tram.
  int sum = 0;
  /**
   * Block Logic: Iterate through each stop to calculate passenger flow.
   * Invariant: Before each iteration, 'sum' holds the number of passengers on the tram as it arrives at the current stop.
   * 'minCap' holds the maximum number of passengers seen so far.
   */
  for (int i = 0; i < n; ++i) {
    // Read the number of passengers exiting and entering at the current stop.
    cin >> a >> b;
    // Update the total number of passengers on the tram.
    // The change is the number of entering passengers minus the number of exiting passengers.
    sum += (b - a);
    // Update the minimum required capacity if the current number of passengers is the highest seen so far.
    minCap = max(minCap, sum);
  }
  // Output the minimum capacity required for the tram.
  cout << minCap;
  return 0;
}
