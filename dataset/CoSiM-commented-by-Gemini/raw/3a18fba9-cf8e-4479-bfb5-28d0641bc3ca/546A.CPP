/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 * @details Calculates the amount of money a soldier needs to borrow to buy a
 *          specific number of bananas, where the cost of each banana increases
 *          linearly.
 *
 * Problem Description:
 * A soldier wants to buy `w` bananas. The first banana costs `k`, the second
 * costs `2k`, and so on, up to the `w`-th banana costing `w*k`. The soldier
 * initially has `n` dollars. The program must calculate the amount of money
 * the soldier needs to borrow. If the soldier has enough money, the amount to
 * borrow is 0.
 *
 * Algorithm:
 * 1. The total cost of `w` bananas is the sum of an arithmetic series:
 *    `k*1 + k*2 + ... + k*w`.
 * 2. This sum can be expressed as `k * (1 + 2 + ... + w)`.
 * 3. The sum of the first `w` positive integers is given by the formula
 *    `w * (w + 1) / 2`.
 * 4. Therefore, the total cost is `k * w * (w + 1) / 2`.
 * 5. The amount to borrow is the `total cost - n` (initial money).
 * 6. If this amount is negative or zero, the soldier doesn't need to borrow,
 *    so the output is 0. Otherwise, the output is the calculated positive amount.
 *
 * Time Complexity: O(1), as the solution involves a direct calculation using a mathematical formula.
 * Space Complexity: O(1).
 *
 * @note The file includes a `isprime` function and other definitions that are
 *       part of a template and are not used in this specific solution.
 */
#include <bits/stdc++.h>
typedef long long ll;
typedef unsigned long long ull;
#define pi 3.141592653589793238462643383279502884197169399375105820974944592307816406286
using namespace std;

// Unused template function for primality testing.
bool isprime(ull n) {
  if (n <= 1)
    return false;
  if (n <= 3)
    return true;
  if (n % 2 == 0 || n % 3 == 0)
    return false;
  for (ull i = 5; i * i <= n; i = i + 6)
    if (n % i == 0 || n % (i + 2) == 0)
      return false;
  return true;
}

int main() {
  // Fast I/O setup.
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);

  long long int k, n, w, b;
  // Read k: cost of the first banana
  // Read n: initial amount of money
  // Read w: number of bananas to buy
  cin >> k >> n >> w;

  // Calculate the total cost using the arithmetic series sum formula.
  b = (k * w * (w + 1) / 2) - n;

  // Determine the amount to borrow.
  // If `b` is negative, the soldier has enough money, so borrow 0.
  if (b < 0)
    cout << "0";
  else
    cout << b;
    
  return 0;
}
