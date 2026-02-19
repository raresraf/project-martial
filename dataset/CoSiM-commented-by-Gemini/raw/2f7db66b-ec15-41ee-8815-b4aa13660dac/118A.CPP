/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem on Codeforces (Problem 118A).
 * @details The program processes a given string according to specific rules:
 *          1. It removes all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 *          2. It inserts a '.' character before each consonant.
 *          3. It converts all consonants to lowercase.
 *
 * Algorithm:
 * 1. Read the input string `s`.
 * 2. Create an empty string `ans` to build the result.
 * 3. Iterate through each character `x` of the input string `s`.
 *    a. Check if the character `x` is a vowel (case-insensitive, including 'Y').
 *    b. If it is a vowel, skip to the next character.
 *    c. If it is a consonant, append a '.' to the `ans` string.
 *    d. Then, append the lowercase version of the consonant `x` to the `ans` string.
 * 4. Print the final `ans` string.
 *
 * Time Complexity: O(L), where L is the length of the input string, because it iterates through the string once.
 * Space Complexity: O(L), to store the original and the resulting strings.
 */
// Link to Problem: https://codeforces.com/problemset/problem/118/A

#include <bits/stdc++.h>

// Common competitive programming macros
#define ll long long
#define ull unsigned long long int
#define ld long double
#define mod 1000000007
#define setbits(x) __builtin_popcount(x)
#define setbits_ll(x) __builtin_popcountll(x)
#define zrobits(x) __builtin_ctz(x)
#define zrobits_ll(x) __builtin_ctzll(x)
#define inf 1e18
#define ps(x, y)                                                               \
  fixed << setprecision(y)                                                     \
        << x //set precision of double x to y decimal places \
                                               //ps(1.254754, 4) -> 1.2547

using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout
   * to optimize I/O performance, which is a common practice in competitive programming.
   */
  ios::sync_with_stdio(false);
  cin.tie(0);

  string s, ans = "";
  cin >> s;
  /**
   * Block Logic: Iterates through the input string to perform the transformation.
   * Pre-condition: `s` contains the input string. `ans` is an empty string.
   * Invariant: `ans` contains the transformed version of the processed prefix of `s`.
   */
  for (int i = 0; i < s.length(); i++) {
    char x = s[i];
    /**
     * Block Logic: Filters out vowels. The problem considers 'a', 'e', 'i', 'o', 'u', and 'y'
     * as vowels, in both uppercase and lowercase.
     */
    if (x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u' || x == 'A' ||
        x == 'E' || x == 'I' || x == 'O' || x == 'U' || x == 'y' || x == 'Y')
      continue;

    // For consonants, prepend a '.' and convert the character to lowercase.
    ans += ".";
    ans += tolower(x);
  }

  cout << ans << endl;
}

/**
 * @brief (Unused) Calculates the power of a number.
 * @details This function computes `a` raised to the power of `b` using exponentiation by squaring.
 *          It is not called anywhere in the main logic of this specific problem.
 * @param a The base.
 * @param b The exponent.
 * @return The result of a^b.
 * @note Time Complexity: O(log b)
 */
int getpow(int a, int b) {
  // Base case: any number to the power of 0 is 1.
  if (b == 0)
    return 1;

  // Recursive step: calculate power of a^(b/2).
  int x = getpow(a, b / 2);
  // If exponent is odd.
  if (b & 1)
    return x * x * a;
  // If exponent is even.
  else
    return x * x;
}
