/**
 * @file 1140D.CPP
 * @brief This program calculates the sum of `i * (i + 1)` for `i` from 2 up to `n-1`.
 * This problem often arises in competitive programming contexts requiring efficient arithmetic series calculation.
 *
 * Algorithm: Iterative summation.
 * 1. Read an integer `n`.
 * 2. Initialize a sum variable `len` to 0.
 * 3. Iterate `i` from 2 to `n-1`, adding `i * (i + 1)` to `len` in each step.
 * 4. Print the final sum `len`.
 * Time Complexity: O(N) where N is the input integer, as the loop runs approximately N times.
 * Space Complexity: O(1) as only a few variables are used.
 */

#include <bits/stdc++.h>
#define endl '
'
#define time clock_t tStart = clock();
#define show                                                                   
  printf("Time taken: %.6fs
", (double)(clock() - tStart) / CLOCKS_PER_SEC);
#define ll long long int
#define loop(a, b) for (int i = a; i <= b; ++i)
#define count_1(n) __builtin_popcountll(n)
#define pb push_back
#define F first
#define S second
#define mp make_pair
#define clr(x) x.clear()
#define MOD 1000000007
#define itoc(c) ((char)(((int)'0') + c))
#define vi vector<int>
#define vll vector<ll>
#define pll pair<ll, ll>
#define pii pair<int, int>
#define all(p) p.begin(), p.end()
#define mid(s, e) (s + (e - s) / 2)
#define sv()                                                                   
  ll t, n;                                                                     
  scanf("%lld", &t);                                                           
  n = t;                                                                       
  while (t--)
#define tcase()                                                                
  ll t, n;                                                                     
  cin >> t;                                                                    
  n = t;                                                                       
  while (t--)
#define iscn(num) scanf("%d", &num);

using namespace std;
void FAST_IO();

/**
 * @brief Main function of the program.
 * Reads an integer `n` and calculates the specific sum.
 * @pre `n` is an integer.
 * @post Prints the calculated sum `len`.
 * @return 0 upon successful execution.
 */
int main() {
  //_time_ // Original commented line, preserved.
  // FAST_IO(); // Original commented line, preserved.

  // Functional Utility: 'n' for the upper bound of the summation, 'len' for the accumulated sum.
  ll n, len = 0;
  // Functional Utility: Reads the input integer `n`.
  cin >> n;
  // Block Logic: Iterates from i = 2 up to n-1 (exclusive of n) to calculate the sum.
  // Invariant: `len` accumulates the sum of `j * (j + 1)` for `j` from 2 up to `i-1`.
  for (ll i = 2; i < n; i++) {
    // Functional Utility: Adds the product `i * (i + 1)` to the running sum.
    len += (i * (i + 1));
  }
  // Functional Utility: Prints the final calculated sum.
  cout << len << endl;
  // show // Original commented line, preserved.
  return 0;
}

/**
 * @brief Optimizes C++ standard streams for faster input/output.
 * This function is typical boilerplate in competitive programming to reduce I/O overhead.
 * It unties cin/cout from stdio and disables synchronization with C stdio.
 */
void FAST_IO() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  // cout.setf(ios::fixed); // Original commented line, preserved.
  // cout.precision(20); // Original commented line, preserved.
#ifndef _offline // Block Logic: Conditional compilation for handling input/output redirection.
  // freopen("input.txt", "r", stdin); // Original commented line, preserved.
  // freopen("output.txt", "w", stdout); // Original commented line, preserved.
#endif
}
