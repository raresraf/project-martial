/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem.
 * @author Pankaj Kumar (INDIA)
 *
 * @details This program calculates the number of problems a team will solve, based on
 * the rule that at least two of the three members must be confident. The code is
 * built upon a large template of macros and typedefs common in competitive programming.
 *
 * @section algorithm Algorithm
 * 1. Fast I/O is enabled using a macro `pan`.
 * 2. The number of problems, `n`, is read from standard input.
 * 3. A `count` variable is initialized to zero to store the number of solvable problems.
 * 4. The code iterates `n` times, once for each problem.
 *    a. A temporary counter `temp` is reset to zero.
 *    b. An inner loop runs three times to read the decisions of the three team members.
 *    c. Inside the inner loop, each decision is read. If it is 1, `temp` is incremented.
 *    d. After reading the three decisions, if `temp` is 2 or more, the main `count` is incremented.
 * 5. Finally, the total `count` is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of problems, as the code iterates through each problem once.
 * - Space Complexity: O(1), as data is processed on the fly without being stored in a large data structure.
 */
#include <bits/stdc++.h>
using namespace std;
// NOTE: The following is a standard template for competitive programming.
typedef long long ll;
typedef set<pair<int, int>> spi;
typedef set<pair<ll, ll>> spl;
typedef vector<pair<int, int>> vpi;
typedef vector<int> vi;
typedef vector<ll> vl;
typedef vector<bool> vb;
typedef vector<char> vc;
typedef vector<pair<ll, ll>> vpl;
typedef vector<string> vs;
typedef map<int, int> mi;
typedef map<ll, ll> ml;
typedef set<string> ss;
typedef set<int> si;
typedef set<ll> sl;
#define pan                                                                    \
  cin.tie(0);                                                                  \
  cout.tie(0);                                                                 \
  ios_base::sync_with_stdio(0);
// define values.
#define mod 1000000007;
#define phi 1.618
/* Bit-Stuff */
#define get_set_bits(a) (__builtin_popcount(a))
#define get_set_bitsll(a) (__builtin_popcountll(a))
#define get_trail_zero(a) (__builtin_ctz(a))
#define get_lead_zero(a) (__builtin_clz(a))
#define get_parity(a) (__builtin_parity(a))
/*  Abbrevations  */
#define ff first
#define ss second
#define mp make_pair
#define line cout << endl;
#define pb push_back
#define Endl "\n"
#define loopi(a, b) for (int i = a; i <= b; i++)
// function

ll power(ll x, ll y) {
  ll res = 1;
  while (y > 0) {
    if (y % 2 == 1)
      res *= x;
    y /= 2;
    x *= x;
  }
  return res;
}
/* ascii value
A=65,Z=90,a=97,z=122
*/
/*  -----------------------------------------------------------------------------------*/

int main() {
  pan; // Macro for fast I/O settings.
  int n, no;
  cin >> n;
  int count = 0, temp = 0;
  // Loop through each of the n problems.
  for (int i = 0; i < n; i++) {
    temp = 0; // Reset confidence counter for the current problem.
    // Inner loop to read decisions from the three members.
    for (int j = 1; j <= 3; j++) {
      cin >> no;
      // If a member is sure (inputs a 1), increment the counter for this problem.
      if (no == 1)
        temp++;
    }
    // If at least two members are sure, the team will solve the problem.
    if (temp >= 2)
      count++;
  }
  cout << count << endl;
}