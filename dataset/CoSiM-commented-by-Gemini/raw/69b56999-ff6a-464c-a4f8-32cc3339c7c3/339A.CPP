
/* Monil Soni sonil06 */
/* DA-IICT  */

#include <bits/stdc++.h>

using namespace std;

/**
 * @file 339A.CPP
 * @brief Solution to the "Helpful Maths" problem from Codeforces (Problem 339A).
 * @details The program takes a string representing a sum of single-digit numbers
 * (e.g., "3+1+2") and rearranges the numbers in non-decreasing order (e.g., "1+2+3").
 *
 * Algorithm:
 * 1. The input sum is read as a string.
 * 2. The numeric characters (summands) are extracted from the string. They are located
 *    at even indices (0, 2, 4, ...).
 * 3. These characters are stored in a vector.
 * 4. The vector of number characters is sorted. Standard character sorting works
 *    correctly as '1' < '2' < '3'.
 * 5. The sorted sum is reconstructed by printing the numbers from the sorted vector,
 *    interspersed with '+' signs.
 *
 * Time Complexity: O(L log L), where L is the number of summands in the string.
 * The complexity is dominated by the sort operation.
 * Space Complexity: O(L) to store the summands in a vector.
 */

typedef long long LL;
typedef vector<int> vi;
typedef vector<long long> vl;

// Macro for a standard for-loop from 0 to n-1.
#define MOD 1000000007
#define _for(i, n) for (int i = 0; i < n; i++)
// Macro for a for-loop from 1 to n.
#define __for(i, n) for (int i = 1; i <= n; i++)
// Macro for a for-loop from a starting value `s` to an ending value `n`.
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++)
// Macro similar to _for, but with a different variable naming convention.
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
// Macro for a reverse for-loop from `e` down to `s`.
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
// Macro for iterating over a container.
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++)
// Shortcut for push_back.
#define pb push_back
// Shortcut for make_pair.
#define mp make_pair
// Shortcut for first element of a pair.
#define fi first
// Shortcut for second element of a pair.
#define se second
// Shortcut for passing the begin and end iterators of a container.
#define ALL(x) x.begin(), x.end()

int main() {
  string str;
  cin >> str;
  int n = str.length();
  vector<char> v;
  
  // Extract the numbers from the string. They are at even indices.
  for (int i = 0; i < n; i += 2) {
    v.pb(str.at(i));
  }
  
  // Sort the numbers.
  sort(v.begin(), v.end());
  
  n = v.size();
  
  // Print the sorted sum.
  printf("%c", v[0]);
  for (int i = 1; i < n; i++) {
    printf("+%c", v[i]);
  }
  printf("\n");
  return 0;
}
