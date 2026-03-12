/* .............
   S A K S H A M
   .............
*/
/* .............
   S A K S H A M
   .............
*/
/**
 * @5949e5f5-5dab-4881-9893-1b9b4c95acee/50B.CPP
 * @brief This program calculates a specific sum based on the frequencies of
 * characters in an input string. For each character in the string, it adds
 * its total count of occurrences (frequency) in the string to a running total.
 * This effectively computes the sum of the squares of the frequencies of all unique characters.
 * The characters can be lowercase letters, uppercase letters, or digits.
 *
 * Algorithm: Two-pass approach: first pass counts character frequencies, second pass sums frequencies.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(C) for the frequency array, where C is the size of the character set (62). Effectively O(1).
 */
#include <bits/stdc++.h> // Includes most standard C++ library headers.
using namespace std;     // Uses the standard namespace.
#define yes cout << "yes\n"; // Macro for printing "yes".
#define no cout << "no\n";   // Macro for printing "no".
#define YES cout << "YES\n"; // Macro for printing "YES".
#define NO cout << "NO\n";   // Macro for printing "NO".
#define ll long long int     // Defines 'll' as a shorthand for long long int.
#define pb push_back         // Defines 'pb' as a shorthand for vector::push_back.
#define mod 1000000007       // Defines 'mod' for a common modulo value.
#define nl cout << "\n";     // Defines 'nl' for printing a newline.
#define ff first             // Defines 'ff' as a shorthand for pair::first.
#define ss second            // Defines 'ss' as a shorthand for pair::second.
#define loop1(m) for (int i = 0; i < m; i++) // Macro for a common for-loop pattern (0 to m-1).
#define vl vector<ll>                     // Defines 'vl' as a shorthand for vector<ll>.
#define vll vector<ll, ll>                // Defines 'vll' as a shorthand for vector<ll, ll> (likely a typo, perhaps vector<pair<ll,ll>>).
#define vc vector<char>                   // Defines 'vc' as a shorthand for vector<char>.
#define msi map<string, int>              // Defines 'msi' as a shorthand for map<string, int>.
#define loop2(n, m) for (auto i = n; i <= m; i++) // Macro for a common for-loop pattern (n to m).
#define count1s(n) __builtin_popcount(n)          // Macro for counting set bits (GCC/Clang specific).

// Utility function to check if a number is a perfect square.
bool psqrt(ll x) {
  long double sr = sqrt(x);
  return ((sr - floor(sr)) == 0);
}
// Utility function to check if a number is odd using bitwise AND.
bool ifodd(ll n) { return (n & 1); }
// Utility function to calculate the least common multiple (LCM).
int lcm(int a, int b) { return a / __gcd(a, b) * b; }
// Utility function to check if a number is prime using a basic primality test.
bool isprime(ll n) {
  if (n <= 1)
    return false;
  if (n <= 3)
    return true;
  if (n % 2 == 0 || n % 3 == 0)
    return false;
  for (int i = 5; i * i <= n; i = i + 6)
    if (n % i == 0 || n % (i + 2) == 0)
      return false;
  return true;
}
int main() {
  // Optimizes C++ standard I/O streams.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  string s; // Declares a string variable to store the input.
  cin >> s; // Reads the input string from standard input.
  ll ans = 0; // Initializes 'ans' to 0, which will store the final calculated sum.
  // Declares a frequency array 'f' of size 62, initialized to zeros.
  // This array will store counts for: 'a'-'z' (0-25), 'A'-'Z' (26-51), '0'-'9' (52-61).
  ll f[62] = {0}; 
  
  /**
   * Block Logic: First pass - iterates through the input string to calculate the frequency
   * of each character (lowercase, uppercase, or digit).
   * Invariant: After this loop, 'f' correctly stores the counts of all relevant characters.
   */
  loop1(s.length()) { // Iterates from i = 0 to s.length() - 1.
    if (s[i] >= 'a' && s[i] <= 'z') // Inline: If the character is a lowercase letter.
      f[s[i] - 'a']++; // Increment its count at index 0-25.
    else if (s[i] >= 'A' && s[i] <= 'Z') // Inline: If the character is an uppercase letter.
      f[s[i] - 'A' + 26]++; // Increment its count at index 26-51.
    else // Inline: Otherwise, assume it's a digit.
      f[s[i] - 48 + 52]++; // Increment its count at index 52-61 ('0' is ASCII 48).
  }
  
  /**
   * Block Logic: Second pass - iterates through the input string again to calculate the final sum.
   * For each character in the string, it adds its total frequency (as found in the first pass) to 'ans'.
   * Invariant: 'ans' accumulates the sum of frequencies for each character encountered.
   */
  loop1(s.length()) { // Iterates from i = 0 to s.length() - 1.
    if (s[i] >= 'a' && s[i] <= 'z') // Inline: If the character is a lowercase letter.
      ans += f[s[i] - 'a']; // Add its frequency to 'ans'.
    else if (s[i] >= 'A' && s[i] <= 'Z') // Inline: If the character is an uppercase letter.
      ans += f[s[i] - 'A' + 26]; // Add its frequency to 'ans'.
    else // Inline: Otherwise, assume it's a digit.
      ans += f[s[i] - 48 + 52]; // Add its frequency to 'ans'.
  }
  cout << ans; // Prints the final calculated sum.
  // The program implicitly returns 0 on successful completion of main.
}
