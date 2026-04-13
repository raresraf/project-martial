/**
 * @file 96A.CPP
 * @brief This program attempts to decompose a given integer 'n' into a sum of 3s, 5s, and 7s.
 * It uses a greedy approach, prioritizing the use of 7s and 5s to reduce 'n',
 * and falling back to 3s when larger denominations are not immediately suitable.
 * The goal is to find the counts of each (c3, c5, c7) that sum up to the original 'n'.
 * If 'n' cannot be reduced to 0 this way, it's considered impossible (-1 output).
 * Algorithm: Greedy decomposition using modulo and subtraction.
 * Time Complexity: O(T * N/3) in the worst case, where T is the number of test cases and N is the initial number to decompose.
 * Space Complexity: O(1) per test case.
 */

#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience.

using namespace std;

typedef long long ll;     // Alias for long long.
typedef unsigned long long ull; // Alias for unsigned long long.

int main() {
  // Optimizes C++ standard streams for faster input/output operations.
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr); // Unties cin from cout.
  cout.tie(nullptr); // Unties cout.

  int T; // Stores the number of test cases.
  // Reads the number of test cases.
  cin >> T;
  /**
   * Block Logic: Processes each of the 'T' test cases.
   * Invariant: Each iteration processes a new integer 'n' for decomposition.
   */
  while (T--) {
    int n; // The integer to be decomposed into 3s, 5s, and 7s.
    // Reads the integer 'n' for the current test case.
    cin >> n;
    int c3 = 0, c5 = 0, c7 = 0; // Counters for the number of 3s, 5s, and 7s used.

    /**
     * Block Logic: Attempts to decompose 'n' into a sum of 3s, 5s, and 7s.
     * It uses a greedy strategy:
     * 1. If 'n' is divisible by 7, it prioritizes using as many 7s as possible.
     * 2. Else, if 'n' is divisible by 5, it prioritizes using as many 5s as possible.
     * 3. Otherwise, it subtracts 3 from 'n' and increments the count of 3s.
     * This loop continues until 'n' becomes 0 or negative.
     * Invariant: 'n' is progressively reduced, and `c3, c5, c7` track the counts.
     */
    while (n > 0) {
      if (n % 7 == 0) { // If 'n' is perfectly divisible by 7, use 7s.
        c7 = n / 7;
        n = 0; // 'n' is fully decomposed.
      } else if (n % 5 == 0) { // Else if 'n' is perfectly divisible by 5, use 5s.
        c5 = n / 5;
        n = 0; // 'n' is fully decomposed.
      } else { // Otherwise, subtract 3 and increment c3.
        n -= 3;
        c3++;
      }
    }

    /**
     * Conditional Logic: Determines the output based on whether 'n' was successfully decomposed to 0.
     * If 'n' is not 0 (meaning it became negative and could not be perfectly decomposed), output -1.
     * Otherwise, output the counts of 3s, 5s, and 7s used.
     */
    if (n) // If n is not 0 (i.e., it's negative, meaning decomposition was impossible).
      cout << "-1\n";
    else // If n is 0 (decomposition was successful).
      cout << c3 << " " << c5 << " " << c7 << "\n";
  }
  // Standard program exit.
  return 0;
}
