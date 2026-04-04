/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem on platforms like Codeforces.
 * @details This program calculates the minimum possible capacity of a tram based on the number of passengers
 * entering and exiting at each stop. It tracks the current number of passengers and finds the maximum
 * number of passengers on the tram at any point in time.
 *
 * Algorithm:
 * 1. Read the number of tram stops, n.
 * 2. Initialize 'total' (current passengers) and 'mtotal' (maximum passengers) to 0.
 * 3. Loop through each of the n stops:
 *    a. Read the number of passengers exiting ('a') and entering ('b').
 *    b. Update the current number of passengers: total = total - a + b.
 *    c. Update the maximum capacity required: mtotal = max(mtotal, total).
 * 4. Print the maximum capacity 'mtotal'.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1), as it only uses a few variables to store the state.
 */
#include <bits/stdc++.h>

using namespace std;
typedef long long ll;
typedef pair<int, int> pii;

int main() {
  int n;
  cin >> n;
  int mtotal = 0, total = 0;
  /**
   * @brief Loop through each stop to calculate passenger changes.
   * Pre-condition: 'n' is the number of stops. 'total' and 'mtotal' are 0.
   * Invariant: After each iteration, 'total' holds the current number of passengers
   * and 'mtotal' holds the maximum number of passengers seen so far.
   */
  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;
    // Update the current number of passengers by subtracting those who exit and adding those who enter.
    total += b - a;
    // Update the maximum required capacity if the current number of passengers is greater.
    mtotal = max(total, mtotal);
  }
  // Output the minimum capacity required for the tram.
  cout << mtotal << endl;
  return 0;
}
