/**
 * @d5e1390a-dffe-4f5f-ae48-06c0347c9c94/546A.CPP
 * @brief Solution for a competitive programming problem involving calculating borrowed money for purchases.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy a certain number of bananas.
 * The cost of bananas increases arithmetically: the first costs 'k', the second '2k', up to 'wk'.
 * The soldier has 'n' dollars initially. The task is to find the deficit, or 0 if no borrowing is needed.
 * This solution uses standard C++ streams with I/O optimizations and directly computes the maximum of
 * zero and the calculated borrowing amount.
 *
 * Algorithm:
 * 1. Read the base cost of a banana 'k', the initial money 'n', and the number of bananas 'w'.
 * 2. Calculate the total cost of buying 'w' bananas by summing `(i + 1) * k` for each banana.
 * 3. Calculate the difference: (total cost - initial money).
 * 4. Print the maximum of 0 and this difference, ensuring the output is never negative.
 *
 * Time Complexity: O(W) due to the loop for summing costs, where W is the number of bananas.
 * Space Complexity: O(1) - uses a fixed amount of memory for variables.
 */

#include <bits/stdc++.h> // Functional Utility: Includes most standard C++ libraries, common in competitive programming.
using namespace std;

typedef long long ll;     // Functional Utility: Defines 'll' as an alias for long long.
typedef pair<int, int> ii; // Functional Utility: Defines 'ii' as an alias for std::pair<int, int>.

int main() {
  // Functional Utility: Optimizes C++ standard I/O for faster execution.
  // Decouples C++ streams from standard C streams and unties cin from cout.
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);

  int k, n, w, ret = 0;
  // Functional Utility: Reads the base cost of one banana 'k', initial money 'n', and number of bananas 'w'.
  cin >> k >> n >> w;
  // Block Logic: Calculates the total cost of 'w' bananas.
  // Pre-condition: 'k', 'n', 'w' are initialized.
  // Invariant: 'ret' accumulates the total cost based on the arithmetic progression.
  for (int i = 0; i < w; i++)
    ret += (i + 1) * k; // Functional Utility: Adds the cost of the (i+1)-th banana to the total.
  // Functional Utility: Calculates the amount to borrow and ensures it's not negative.
  // Uses std::max to return 0 if 'ret - n' is negative, otherwise 'ret - n'.
  cout << max(0, ret - n) << endl;
  return 0; // Functional Utility: Indicates successful program execution.
}
