
#include <bits/stdc++.h>

/**
 * @file 281A.CPP
 * @brief Solution to Codeforces problem 281A, "Word Capitalization".
 * @details The program takes a single word as input and capitalizes its first
 * letter without changing the rest of the word.
 *
 * Algorithm:
 * The program reads the entire word into a string. It then accesses the first
 * character of the string (at index 0) and converts it to its uppercase
 * equivalent using the `toupper()` function. The modified string is then
 * printed to the console.
 *
 * Time Complexity: O(L), where L is the length of the string, due to reading
 * and printing the string. The capitalization itself is an O(1) operation.
 * Space Complexity: O(L) to store the input string.
 */

// --- Boilerplate and macros for competitive programming (mostly unused) ---
#define ll long long int
#define loop(k) for (i = 0; i < k; ++i)
#define loop2(k, l) for (j = k; j < l; ++j)
#define mod 1000000007
using namespace std;

int main() {
  // Fast I/O setup.
  std::ios_base::sync_with_stdio(false); // cin.tie(NULL);
  
  // Unused variables common in competitive programming templates.
  ll n = 1, i, j, t;
  
  string s;
  cin >> s;
  
  // Capitalize the first character of the string.
  s[0] = toupper(s[0]);
  
  cout << s;
  cout << "\n";
  return 0;
}
