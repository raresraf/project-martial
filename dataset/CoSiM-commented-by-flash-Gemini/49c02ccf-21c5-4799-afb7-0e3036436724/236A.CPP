
/**
 * @49c02ccf-21c5-4799-afb7-0e3036436724/236A.CPP
 * @brief This program determines an output message based on the parity of
 *        the number of unique characters in an input string.
 * Algorithm: Uses `std::set` to efficiently count unique characters, then checks parity.
 * Time Complexity: O(L log L) where L is the length of the input string, due to insertion into a set.
 * Space Complexity: O(C) where C is the number of unique characters (at most 26 for English alphabet).
 */

#include <bits/stdc++.h>

using namespace std;

typedef long long LL; // Type definition for long long integers.
typedef vector<int> vi; // Type definition for a vector of integers.
typedef vector<long long> vl; // Type definition for a vector of long long integers.

#define MOD 1000000007 // Macro for a modulo constant, often used in competitive programming.
#define _for(i, n) for (int i = 0; i < n; i++) // Macro for a common for loop pattern (0 to n-1).
#define __for(i, n) for (int i = 1; i <= n; i++) // Macro for a common for loop pattern (1 to n).
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++) // Macro for a for loop with a custom start and end.
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++) // Macro for a for loop (0 to n-1).
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--) // Macro for a reverse for loop.
// Macro for iterating through a container. 'typeof' is a GNU extension.
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++)
#define pb push_back // Macro for `push_back` operation on vectors.
#define mp make_pair // Macro for `make_pair` operation for pairs.
#define fi first     // Macro for accessing the first element of a pair.
#define se second    // Macro for accessing the second element of a pair.
#define ALL(x) x.begin(), x.end() // Macro for getting iterators to the beginning and end of a container.

int main() {
  string str; // Declare a string variable 'str' to store the input username.
  // Functional Utility: Read the input username string.
  cin >> str;
  // Block Logic: Sort the string to bring identical characters together.
  sort(str.begin(), str.end());
  int n = str.size(); // Store the length of the string.
  set<char> s; // Declare a set of characters to store unique characters.
  // Block Logic: Iterate through the sorted string and insert each character into the set.
  // A set automatically handles uniqueness.
  for (int i = 0; i < n; i++) {
    s.insert(str.at(i));
  }
  // Block Logic: Check the parity of the count of unique characters to determine the output message.
  if (s.size() % 2 == 0) // If the number of unique characters is even.
    printf("CHAT WITH HER!\n");
  else // If the number of unique characters is odd.
    printf("IGNORE HIM!\n");
  return 0; // Indicate successful program termination.
}
