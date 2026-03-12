/**
 * @61b5b880-30fb-4c3f-9d45-7531e687b310/1221C.CPP
 * @brief This program solves Codeforces problem 1221C, "Perfect Team".
 * It determines the maximum number of complete teams that can be formed
 * given `c` programmers, `m` mathematicians, and `x` generalists.
 * A team requires one programmer, one mathematician, and one generalist.
 * This solution uses a direct formula that accounts for the limiting factors
 * of total available people and the minimum number of specialists.
 *
 * Algorithm: Direct mathematical calculation involving minimums of total resources and specialist counts.
 * Time Complexity: O(T) where T is the number of test cases.
 * Space Complexity: O(1) for storing variables.
 */

#include <bits/stdc++.h> // Includes many standard C++ library headers.
#define ll long long int // Defines 'll' as a shorthand for long long int.
#define pb push_back     // Defines 'pb' as a shorthand for vector::push_back.
#define F first          // Defines 'F' as a shorthand for pair::first.
#define S second         // Defines 'S' as a shorthand for pair::second.
#define mp make_pair     // Defines 'mp' as a shorthand for std::make_pair.
#define MOD 1000000007   // Defines 'MOD' for a common modulo value.
#define vi vector<int>   // Defines 'vi' as a shorthand for vector<int>.
#define vll vector<ll>   // Defines 'vll' as a shorthand for vector<ll>.
#define pll pair<ll, ll> // Defines 'pll' as a shorthand for pair<ll, ll>.
#define pii pair<int, int> // Defines 'pii' as a shorthand for pair<int, int>.
#define all(p) p.begin(), p.end() // Defines 'all' for iterator range.
#define mid(s, e) (s + (e - s) / 2) // Defines 'mid' for calculating middle index.
#define eb emplace_back // Defines 'eb' as a shorthand for vector::emplace_back.
#define ull unsigned long long // Defines 'ull' as a shorthand for unsigned long long.
#define KAMEHAMEHA ios_base::sync_with_stdio(0); // Macro for fast I/O (same as RASENGAN functionally).
#define RASENGAN                                                               \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
using namespace std; // Uses the standard namespace.
int main() {
  KAMEHAMEHA // Calls the fast I/O macro.
  //    #ifdef _soumik // Commented out conditional compilation for input redirection.
  //        freopen("input.txt", "r", stdin);
  //    #endif
  int q, c, m, x; // Declares variables: q (test cases), c (programmers), m (mathematicians), x (generalists).
  cin >> q;       // Reads the number of test cases.
  /**
   * Block Logic: Processes each test case to calculate the maximum number of teams.
   * Invariant: For each test case, the maximum number of teams is calculated and printed.
   */
  while (q--) { // Loops 'q' times, once for each test case.
    cin >> c >> m >> x; // Reads 'c', 'm', and 'x' for the current test case.
    int mn = min(c, m); // Inline: Calculates the minimum count between programmers and mathematicians.
                        // This represents one upper bound for the number of teams, as each team needs one of each.
    // int extra=c+m+x-(2*mn); // Commented out: suggests an alternative calculation for 'extra' generalists.
    // Block Logic: The core calculation for the maximum number of teams.
    // It's the minimum of two limiting factors:
    // 1. `(c + m + x) / 3`: Represents the maximum teams if all individuals (programmers, mathematicians, generalists)
    //    could be perfectly combined into teams of three, ignoring specialist roles initially.
    // 2. `mn`: Represents the maximum teams limited by the availability of the less numerous specialist (programmer or mathematician).
    // The final answer is the most restrictive of these two limits.
    cout << min((c + m + x) / 3, mn) << endl; // Prints the calculated maximum number of teams.
  }
  return 0; // Indicates successful program termination.
}
