/**
 * @file 96A.CPP
 * @brief Solves a variation of the change-making problem for denominations {3, 5, 7}.
 *
 * @details This program determines the number of 3s, 5s, and 7s required to sum up to a given
 * integer `n`. It uses a greedy approach, prioritizing the use of 7s and 5s
 * when the remaining amount is a direct multiple of them. If not, it subtracts 3
 * and continues, effectively exploring combinations.
 *
 * Algorithm: Greedy with decrementing.
 * Time Complexity: O(T * N), where T is the number of test cases and N is the target value.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef unsigned long long ull;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from C stdio and unties cin from cout
   * to accelerate I/O operations, a common practice in competitive programming.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);

  int T;
  cin >> T;
  // Block Logic: Process multiple independent test cases.
  while (T--) {
    int n;
    cin >> n;
    int c3 = 0, c5 = 0, c7 = 0;

    /**
     * Block Logic: Greedily constructs the sum.
     * Pre-condition: `n` is the remaining value to be formed.
     * Invariant: In each iteration, `n` is reduced by making a greedy choice.
     * The strategy is to check if the remainder can be perfectly divided by 7 or 5.
     * If not, it decrements the total by 3, hoping to reach a state
     * where it becomes divisible.
     */
    while (n > 0) {
      if (n % 7 == 0)
        c7 = n / 7, n = 0;
      else if (n % 5 == 0)
        c5 = n / 5, n = 0;
      else
        n -= 3, c3++;
    }

    /**
     * Block Logic: Check for a valid solution.
     * If `n` is non-zero, it means the loop terminated because `n` became negative,
     * indicating that the number could not be formed with the given denominations.
     */
    if (n)
      cout << "-1
";
    else
      cout << c3 << " " << c5 << " " << c7 << "
";
  }
}
