/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem on Codeforces.
 *
 * @details This program calculates the minimum required capacity of a tram. It does so
 * by first storing the number of passengers inside the tram after each stop in an
 * array, and then iterating through this array to find the maximum value.
 *
 * Algorithm:
 * 1. Read the number of stops, n.
 * 2. Create an array `d` to store the passenger count at each stop.
 * 3. Handle the first stop separately: read exit/entry counts and store the initial
 *    number of passengers in `d[0]`.
 * 4. Loop from the second stop to the last (i=1 to n-1):
 *    a. Read exit ('x') and entry ('y') counts for the current stop.
 *    b. Calculate the passenger count as `d[i] = d[i-1] + y - x`.
 * 5. After populating the array `d`, iterate through it to find the maximum value.
 * 6. Print the maximum value, which represents the minimum required tram capacity.
 *
 * Time Complexity: O(N) for the two separate loops.
 * Space Complexity: O(N) to store the passenger counts in the array 'd'.
 */
#include <iostream>

using namespace std;

// Array to store the number of passengers at each of the 'n' stops.
long long d[11111];

int main() {
  int n;
  cin >> n;
  int x, y;
  // Handle the first stop. Assumes tram starts empty, so passengers = y.
  cin >> x >> y;
  d[0] = y;
  long long ans = 0;
  /**
   * @brief Loop to calculate and store passenger count at each subsequent stop.
   * Pre-condition: d[0] is initialized with the passenger count after the first stop.
   * Invariant: At the end of each iteration 'i', d[i] holds the total number of
   * passengers on the tram after stop 'i+1'.
   */
  for (int i = 1; i < n; ++i) {
    cin >> x >> y;
    d[i] = d[i - 1] + y - x;
  }
  /**
   * @brief Find the maximum value in the passenger count array.
   * This determines the peak number of passengers and thus the minimum
   * required capacity.
   */
  for (int i = 0; i < n; ++i)
    ans = max(ans, d[i]);
  cout << ans;
}