/**
 * @file 898B.CPP
 * @brief Solves a linear Diophantine equation `ax + by = n`.
 *
 * This program attempts to find non-negative integer solutions (x, y) for the
 * equation `a*x + b*y = n`, where n, a, and b are given. It's a common problem
 * in competitive programming related to buying items with different costs to
 * reach an exact total.
 *
 * Algorithm:
 * 1. Read the target sum 'n' and the costs 'a' and 'b'.
 * 2. Check for trivial solutions where only one type of item is used (i.e.,
 *    if 'n' is divisible by 'a' or 'b').
 * 3. Iteratively check for solutions by fixing the number of 'a' items (i)
 *    and checking if the remainder `n - a*i` is divisible by 'b'.
 * 4. If no solution is found, iteratively check by fixing the number of 'b'
 *    items (i) and checking if the remainder `n - b*i` is divisible by 'a'.
 * 5. If a solution is found at any point, print "YES" and the counts, then exit.
 * 6. If all checks fail, print "NO".
 *
 * @note The second search loop contains a bug in its output statement, where it
 *       divides the remainder by 'b' instead of 'a'.
 *
 * Time Complexity: O(n/a + n/b) in the worst case, but bounded by the loop
 * limit of 10,000,000. For typical inputs, it's effectively O(n/min(a,b)).
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
long long n, a, b;
int main() {
  cin >> n;
  cin >> a >> b;

  // Case 1: Check for a solution using only 'a' items.
  if (n % a == 0)
    return cout << "YES\n" << n / a << " " << 0, 0;
  
  // Case 2: Check for a solution using only 'b' items.
  if (n % b == 0)
    return cout << "YES\n" << 0 << " " << n / b, 0;

  /**
   * Block Logic: Iteratively search for a solution by fixing the count of 'a'.
   * 'i' represents the number of items of cost 'a' (the 'x' in ax+by=n).
   */
  for (int i = 1; i < 10000000; i++) {
    long long x = n - (a * i);
    // Check if the remaining amount 'x' is non-negative and can be formed by 'b's.
    if (x % b == 0 && x >= 0)
      return cout << "YES\n" << i << " " << (x == 0 ? 0 : x / b), 0;
  }

  /**
   * Block Logic: Iteratively search for a solution by fixing the count of 'b'.
   * 'i' represents the number of items of cost 'b' (the 'y' in ax+by=n).
   * @note Bug: The output for the number of 'a' items is `x / b`, but it should be `x / a`.
   */
  for (int i = 1; i < 10000000; i++) {
    long long x = n - (b * i);
    // Check if the remaining amount 'x' can be formed by 'a's.
    if (x % a == 0 && x >= 0)
      return cout << "YES\n" << (x == 0 ? 0 : x / b) << " " << i, 0;
  }
  
  // If no solution is found after all checks, print "NO".
  cout << "NO";
  return 0;
}
