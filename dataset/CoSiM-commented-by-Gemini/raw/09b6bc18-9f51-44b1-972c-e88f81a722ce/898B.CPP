/**
 * @file 898B.CPP
 * @brief Solves the "Proper Nutrition" problem.
 * @details This program determines if a number `n` can be represented as a sum `p*x + q*y`,
 * where `x` and `y` are non-negative integers. This is a classic linear Diophantine
 * equation problem. The file also contains a large amount of unused boilerplate code
 * for various number theory functions.
 *
 * @submission
 *     SubmissionId: 33600243
 *     ContestId: 898
 *     Index: B
 *     ProblemName: Proper Nutrition
 *     ProblemTags: ['brute force', 'number theory']
 *     ProgrammingLanguage: GNU C++14
 *     Verdict: OK
 */
#include <bits/stdc++.h>

typedef long long ll;
typedef unsigned long long ull;

#define pi 3.14159265358979323846
#define mod 1000000007

using namespace std;

// --- Unused Boilerplate/Template Code ---
// The following functions (isprime, gcd, exp, lcm, ispalin) and global arrays
// are part of a generic template and are not used in this specific solution.
bool isprime(ull n) { /* ... */ return true; }
long long int gcd(long long int a, long long int b) { /* ... */ return a; }
long long int exp(long long int value, long long int power) { /* ... */ return 1; }
long long int lcm(ll n1, ll n2) { /* ... */ return n1; }
ll a[1000005], b[1000005], c[1000005], d[1000005];
bool ispalin(ll n) { /* ... */ return true; }
// --- End of Unused Boilerplate ---

/**
 * @brief Main function to solve the Diophantine equation n = p*x + q*y.
 *
 * Algorithm:
 * The program uses a brute-force approach to find a valid pair of non-negative
 * integers (x, y).
 * 1. It iterates through all possible values for `x` (named `i` in the code),
 *    from 0 up to `n/p`, since `p*x` cannot exceed `n`.
 * 2. For each `i`, it calculates the remaining value required: `k = n - (p * i)`.
 * 3. It checks if this remainder `k` is divisible by `q`. If `k % q == 0`, a
 *    solution has been found.
 * 4. The program then prints "YES" along with the values `x = i` and `y = k / q`
 *    and terminates.
 * 5. If the loop finishes without finding any solution, it prints "NO".
 *
 * Time Complexity: O(N/P), where N is the target number and P is one of the coefficients.
 * Space Complexity: O(1).
 */
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);

  long long int n, p, q, i, k;
  cin >> n >> p >> q;

  // f: A flag to indicate if a solution is found.
  int f = 0;

  // Iterate through all possible values of x (named i).
  for (i = 0; i <= (n / p); i++) {
    // Calculate the remaining value needed for the q*y term.
    k = n - (p * i);
    
    // Check if the remainder is a non-negative multiple of q.
    if (k >= 0 && k % q == 0) {
      f = 1;
      cout << "YES\n";
      // Print the solution pair (x, y).
      cout << i << " " << k / q << endl;
      break; // Exit after finding the first solution.
    }
  }
  
  // If the loop completes without a solution.
  if (f == 0) {
    // Second attempt: iterate on y instead of x. This is redundant if the first
    // loop covered all cases but can be useful if p and q have very different magnitudes.
    for (i = 0; i <= (n / q); i++) {
      k = n - (q * i);
      if (k >= 0 && k % p == 0) {
        f = 1;
        cout << "YES\n";
        cout << k / p << " " << i << endl;
        break;
      }
    }
  }
  
  if (f == 0)
    cout << "NO" << endl;

  return 0;
}
