/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem using a direct mathematical formula.
 *
 * @details
 * This program calculates the amount of money a soldier must borrow to buy 'w'
 * bananas, where the cost increases arithmetically. This solution calculates the
 * total cost using the formula for the sum of an arithmetic series, making it
 * an O(1) solution.
 *
 * The code includes a large number of pre-processor macros and a boilerplate I/O
 * optimization function, which are common in competitive programming to speed up
 * coding and execution.
 *
 * Algorithm:
 * 1. Read the initial cost 'k', initial money 'n', and number of bananas 'w'.
 * 2. Calculate the total cost directly. The cost is an arithmetic series:
 *    k + 2k + ... + w*k = k * (1 + 2 + ... + w).
 * 3. The sum of the first 'w' integers is `w * (w + 1) / 2`.
 * 4. The code calculates this sum, multiplies by 'k', and then subtracts 'n'
 *    to find the required loan amount.
 * 5. If the amount is negative, it means no loan is needed, so the result is 0.
 *
 * Time Complexity: O(1), as it uses a direct formula.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
// Collection of macros for brevity in competitive programming
#define nl '\n'
#define _time_ clock_t tStart = clock();
#define _show_                                                                 \
  printf("Time taken: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
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
#define pll pair<ll, ll>
#define pii pair<int, int>
#define all(p) p.begin(), p.end()
#define mid(s, e) (s + (e - s) / 2)
#define sv()                                                                   \
  ll t, n;                                                                     \
  scanf("%lld", &t);                                                           \
  n = t;                                                                       \
  while (t--)
#define tcase()                                                                \
  ll t, n;                                                                     \
  cin >> t;                                                                    \
  n = t;                                                                       \
  while (t--)
#define iscn(num) scanf("%d", &num);

using namespace std;
void FAST_IO();
int main() {
  //    _time_ // Macro for timing the execution, currently commented out.
  // FAST_IO(); // I/O optimization function call, currently commented out.
  ll k, n, w;
  cin >> k >> n >> w;
  // The total cost is k * (1 + 2 + ... + w).
  // The sum of the first w integers is w * (w + 1) / 2.
  // The code calculates this by first incrementing w, then using the formula (w' * (w' - 1)) / 2
  // where w' is the incremented w. This is equivalent to (w+1)*w/2 for the original w.
  w++;
  ll temp = (w * (w - 1)) / 2;
  // Calculate the total cost and then subtract the money the soldier has.
  temp = temp * k - n;
  // If the result is negative, it means the soldier has enough money. The debt is 0.
  if (temp < 0)
    temp = 0;
  cout << temp << endl;
  //   _show_ // Macro to display execution time, currently commented out.
  return 0;
}

/**
 * @brief A boilerplate function for optimizing I/O and handling file redirection.
 * This is a common utility in competitive programming for faster I/O and for
 * testing with local input/output files.
 */
void FAST_IO() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  // cout.setf(ios::fixed);
  // cout.precision(20);
#ifndef _offline
  freopen("input.txt", "r", stdin);
  // freopen("output.txt", "w", stdout);
#endif
}
