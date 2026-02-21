/*
        SubmissionId	:	17336820
        ContestId	:	546
        Index	:	A
        ProblemName	:	Soldier and Bananas
        ProblemTags	:	['implementation', 'math']
        ProgrammingLanguage	:	GNU C++11
        Verdict	:	OK
*/
/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem (Codeforces 546A).
 *
 * @details This program calculates the amount of money a soldier needs to borrow.
 * The soldier buys `w` bananas, where the cost of the i-th banana is `i * k`.
 * The program calculates the total cost, subtracts the soldier's initial money `n`,
 * and outputs the difference if it's positive, otherwise 0.
 *
 * @algorithm The core of the problem is to sum an arithmetic series. The total cost is
 * `k + 2k + ... + w*k = k * (1 + 2 + ... + w)`.
 * The program uses the mathematical formula for the sum of the first `w` integers,
 * which is `w * (w + 1) / 2`. This allows calculating the total cost in O(1) time.
 * The amount to borrow is then `TotalCost - n`.
 *
 * @complexity
 * Time Complexity: O(1), as the solution is a direct arithmetic formula.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
typedef long long ll;
typedef unsigned long long ull;
// Unused constant for PI.
#define pi                                                                     \
  3.141592653589793238462643383279502884197169399375105820974944592307816406286
using namespace std;
// Unused primality test function from a template.
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
  cin >> k >> n >> w; // k: cost of first banana, n: initial money, w: number of bananas.
  
  // Calculate the total cost using the formula for an arithmetic series sum.
  b = (k * w * (w + 1) / 2) - n;
  
  // If the cost is less than or equal to the money on hand, no borrowing is needed.
  if (b < 0)
    cout << "0";
  else
    cout << b;
  return 0;
}
