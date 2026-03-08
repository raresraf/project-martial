/**
 * @file 753A.CPP
 * @brief Solution for the "Santa Claus and Candies" problem (e.g., Codeforces 753A).
 * @details The problem is to distribute 'n' candies among the maximum possible number of
 * children, such that each child receives a distinct number of candies (1, 2, 3, ...).
 * This solution uses a greedy approach.
 *
 * @algorithm Greedy Algorithm with Arithmetic Series Summation.
 * To maximize the number of children, we give the smallest possible distinct amounts
 * first (1, 2, 3, ...). The core of the solution is to find the largest 'k' such that
 * the sum 1 + 2 + ... + k does not exceed 'n'.
 * @complexity
 * - Time: O(sqrt(n)). The loop to find the number of children 'i' runs until
 *   `i * (i + 1) / 2 > n`, which means `i` is approximately `sqrt(2n)`.
 * - Space: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

// --- Common Competitive Programming Macros & Typedefs ---
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define REP(i, a, b) for (int i = a; i < b; i++)
typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pi;
#define llp 1000000007 // Unused macro

int n; // Total number of candies.

int main() {
  ios::sync_with_stdio(false); // Fast I/O
  cin.tie(NULL); // Untie cin from cout

  cin >> n;
  int i;
  int k = 0; // Sum of candies for 'i' children.

  /**
   * @block Find Maximum Number of Children
   * @logic This loop finds the largest number of children ('i') we can give
   * 1, 2, ..., i candies to without exceeding the total 'n'.
   * It uses the arithmetic series sum formula: k = i * (i + 1) / 2.
   */
  for (i = 1; ; i++) {
    k = i * (i + 1) / 2;
    if (k > n)
      break;
  }
  
  // After the loop, 'i' is the first number of children for which distribution fails.
  // So, the maximum number of children is 'i - 1'.
  i--;
  
  cout << i << "
"; // Print the maximum number of children.

  /**
   * @block Print Candy Distribution
   * @logic The first `i-1` children get 1, 2, ..., `i-1` candies.
   * The last child gets all the remaining candies to ensure the sum is exactly 'n'.
   */
  REP(j, 1, i) { // Print candies for the first i-1 children.
    cout << j << " ";
  }

  // Calculate candies for the last child: n - (sum of candies for first i-1 children).
  // Sum = (i-1) * i / 2. Since `i` was already decremented, the formula becomes i * (i-1) / 2.
  cout << (n - (i * (i - 1) / 2)) << "
";
  
  return 0;
}
