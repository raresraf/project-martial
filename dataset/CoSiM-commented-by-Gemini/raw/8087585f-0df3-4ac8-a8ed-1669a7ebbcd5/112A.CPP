/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * The program reads two strings, converts them to lowercase in-place, and then
 * compares them lexicographically. It outputs -1 if the first string is smaller,
 * 1 if the second string is smaller, and 0 if they are equal.
 *
 * @algorithm In-place lowercase conversion followed by direct comparison.
 * @complexity
 * Time: O(N), where N is the length of the strings. The process involves a single
 *       pass for converting to lowercase and another for comparison.
 * Space: O(1) auxiliary space, as the conversion happens in-place on the
 *        input strings.
 */
#include <bits/stdc++.h> // Includes all standard libraries, common in competitive programming
using namespace std;

// --- Competitive Programming Boilerplate ---
// A large number of typedefs and macros for shorthand, typical in competitive programming
// to speed up coding. These do not affect the core logic of the solution.
typedef long long LL;
typedef pair<int, int> pii;
typedef pair<LL, LL> pll;
typedef pair<string, string> pss;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<pii> vii;
typedef vector<LL> vl;
typedef vector<vl> vvl;
// ... (rest of boilerplate omitted for brevity in this comment)
#define FOR(a, b, c) for (int(a) = (b); (a) < (c); ++(a))
#define FAST_INP                                                               
  ios_base::sync_with_stdio(false);                                            
  cin.tie(NULL)

/**
 * @brief Main function to handle input, conversion, and comparison.
 */
int main() {
  // FAST_INP is a macro to speed up C++ I/O operations.
  FAST_INP;
  string a, b;
  cin >> a >> b;

  // Pre-condition: 'a' and 'b' contain the input strings.
  // Block Logic: Iterate through the strings to convert all characters to lowercase in-place.
  FOR(i, 0, a.length()) {
    // The condition `a[i] < 92` is a way to check for uppercase letters in ASCII.
    // 'A'-'Z' are 65-90, while 'a'-'z' are 97-122. 92 is the backslash character ''.
    // This check effectively separates uppercase from lowercase letters.
    if (a[i] < 92)
      a[i] += 32; // Add 32 to convert from uppercase to lowercase in ASCII.
    if (b[i] < 92)
      b[i] += 32;
  }

  // Invariant: Both strings 'a' and 'b' are now entirely in lowercase.
  // Block Logic: Perform a standard lexicographical comparison and print the result.
  if (a < b)
    cout << -1;
  else if (a > b)
    cout << 1;
  else if (a == b)
    cout << 0;
  
  return 0; // Added for completeness.
}
