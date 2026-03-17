/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem.
 *
 * @details This program calculates the minimum possible capacity of a tram by finding the
 * maximum number of passengers inside the tram at any point in its route. It simulates
 * the change in passenger count at each stop.
 *
 * @section
 * Algorithm:
 * 1. Read the number of stops, n.
 * 2. Initialize a running count of current passengers to 0.
 * 3. For each stop:
 *    a. Read the number of passengers exiting and entering.
 *    b. Update the current passenger count.
 *    c. Store the new passenger count in a vector.
 * 4. After all stops are processed, sort the vector of passenger counts in descending order.
 * 5. The first element of the sorted vector is the maximum passenger count, which is the required capacity.
 *
 * Time Complexity: O(N log N), where N is the number of stops. The complexity is dominated by the sorting operation.
 * Space Complexity: O(N), to store the passenger counts at each stop.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, temp = 0;
  scanf("%d", &n);
  int a, b;
  vector<int> v;
  /**
   * @brief This loop simulates the tram's journey through 'n' stops.
   * Logic: In each iteration, it updates the current number of passengers ('temp')
   * based on those exiting ('a') and entering ('b'), and records the new total.
   */
  for (int i = 0; i < n; i++) {
    scanf("%d%d", &a, &b);
    temp = temp - a + b;
    v.push_back(temp);
  }

  // Block Logic: Sorts the recorded passenger counts to find the maximum value.
  // An ascending sort followed by a reverse results in a descendingly sorted vector.
  sort(v.begin(), v.end());
  reverse(v.begin(), v.end());

  // The first element of the descendingly sorted vector is the maximum passenger count.
  cout << v[0] << "\n";

  return 0;
}
