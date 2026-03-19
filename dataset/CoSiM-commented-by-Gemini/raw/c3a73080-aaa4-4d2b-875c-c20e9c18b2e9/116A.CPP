/**
 * @file 116A.CPP
 * @brief Inefficient solution for the "Tram" problem (Codeforces 116A).
 * @details This program calculates the minimum required capacity of a tram.
 * It does so by recording the passenger count at every stop and then finding
 * the maximum among these recorded counts.
 *
 * Algorithm: Store and Sort
 * The program simulates the passenger flow, but instead of just tracking the peak
 * value in a single variable, it stores the passenger count after each stop in
 * a `std::vector`. After all stops have been processed, it sorts the entire
 * vector and prints the last element, which corresponds to the maximum passenger
 * count.
 *
 * NOTE: This approach is functionally correct but inefficient.
 * Time Complexity: O(N log N), dominated by the sorting operation, where N is the number of stops.
 * Space Complexity: O(N), due to storing all intermediate passenger counts in the vector.
 * A more optimal solution could achieve this in O(N) time and O(1) space.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // a: number of stops.
  // b: passengers exiting.
  // c: passengers entering.
  // i, j, l, k, o: unused variables.
  // e: current number of passengers.
  int a, b, c, i, j, l, k, o = 0, e = 0;
  
  cin >> a;
  
  // d: a vector to store the passenger count at each stop.
  vector<int> d;

  /**
   * Block Logic: Simulate passenger flow and store each state.
   * Pre-condition: 'a' holds the number of stops to simulate.
   */
  while (a--) {
    cin >> b;
    cin >> c;
    
    // Update the current passenger count.
    e = e - b + c;
    // Store the current count in the vector.
    d.push_back(e);
  }
  
  // Sort the vector of passenger counts in ascending order.
  sort(d.begin(), d.end());
  
  // The last element of the sorted vector is the maximum value, which is the required capacity.
  cout << d[d.size() - 1];
}
