/**
 * @file 236A.CPP
 * @brief Solution attempting to determine a social outcome based on string character uniqueness.
 *
 * This program processes an input string (username) and, based on the parity of a calculated
 * count related to unique characters, prints one of two messages: "CHAT WITH HER!" or "IGNORE HIM!".
 *
 * Algorithm: Character frequency counting using a boolean array.
 * Time Complexity: O(L + C), where L is the length of the input string and C is the size of the
 *                  character set considered (implicitly 26 for lowercase English letters, plus additional array checks).
 * Space Complexity: O(C) for the boolean frequency array.
 */
#include <bits/stdc++.h> // Common header in competitive programming, includes most standard libraries for convenience.

// Macros for common operations and type definitions, frequently used in competitive programming for brevity.
#define nl '\n'      ///< @brief Macro for newline character, often used instead of std::endl for performance.
#define ll long long int ///< @brief Type definition for long long integer.
#define loop(i, a, b) for (int i = a; i < b; ++i) ///< @brief Macro for a common for-loop pattern (from 'a' to 'b'-1).
#define count_1(n) __builtin_popcountll(n) ///< @brief Built-in function to count set bits in a long long integer.
#define pb push_back   ///< @brief Macro for std::vector::push_back.
#define F first        ///< @brief Macro for accessing the first element of a pair.
#define S second       ///< @brief Macro for accessing the second element of a pair.
#define mp make_pair   ///< @brief Macro for std::make_pair.
#define clr(x) x.clear() ///< @brief Macro to clear a container.
#define MOD 1000000007 ///< @brief Constant for modulo operation, frequently used in problems involving large numbers.
#define itoc(c) ((char)(((int)'0') + c)) ///< @brief Converts an integer digit to its character representation.
#define vi vector<int> ///< @brief Type definition for a vector of integers.
#define pll pair<ll, ll> ///< @brief Type definition for a pair of long long integers.
#define pii pair<int, int> ///< @brief Type definition for a pair of integers.
#define all(p) p.begin(), p.end() ///< @brief Macro for getting iterators to the beginning and end of a container.
#define max(x, y) (x > y) ? x : y ///< @brief Macro for finding the maximum of two values.
#define min(x, y) (x < y) ? x : y ///< @brief Macro for finding the minimum of two values.
#define mid(s, e) (s + (e - s) / 2) ///< @brief Macro for calculating the middle index.
// Macro for handling multiple test cases with scanf.
#define tcase()                                                                \
  ll t;                                                                        \
  scanf("%lld", &t);                                                           \
  while (t--)
#define iscn(num) scanf("%d", &num); ///< @brief Macro for scanning an integer using scanf.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

/**
 * @brief Function to optimize C++ standard I/O streams.
 * Disables synchronization between C-style I/O and C++ streams, and unties cin/cout,
 * which can significantly speed up I/O operations in competitive programming.
 */
void FAST_IO();

// Global boolean array to track the presence of characters.
// Sized for 29, implicitly covering 'a' through 'z' (indices 0-25),
// with indices 26-28 always remaining false.
bool alphabet[29];

/**
 * @brief Main function: entry point of the program.
 * Reads a username, determines a count based on character presence, and prints
 * a specific message ("CHAT WITH HER!" or "IGNORE HIM!") based on the parity of this count.
 */
int main() {
  FAST_IO(); // Optimizes I/O.
  string name; // Declares a string 'name' to store the input username.
  cin >> name; // Reads the username.

  int cnt = 0; // Initializes a counter 'cnt'.

  /**
   * @brief Marks the presence of lowercase English letters in the 'alphabet' array.
   * Block Logic: Iterates through the input 'name' string. For each character,
   * it calculates an index ('a' corresponds to 0, 'b' to 1, etc.) and sets the
   * corresponding position in the 'alphabet' array to true.
   * Pre-condition: 'name' contains the input string. 'alphabet' is initialized to all false.
   */
  for (int i = 0; i < name.size(); i++)
    // Inline: Maps a character 'c' to an integer index (c - 'a') for array access.
    alphabet[name[i] - 'a'] = 1;

  /**
   * @brief Counts characters that are NOT present in the 'alphabet' array.
   * Block Logic: Iterates through the first 29 potential character positions (0-28)
   * in the 'alphabet' array. Increments 'cnt' for each position that is false,
   * indicating the absence of the corresponding character.
   *
   * Note on interpretation: For the typical "Boy or Girl" problem (Codeforces 236A),
   * one would count the *distinct characters present*. This code counts
   * (26 - distinct_characters_present) + 3, due to the array size and loop bounds.
   * Let D be the count of distinct lowercase English characters present in 'name'.
   * The calculated 'cnt' effectively becomes (29 - D).
   */
  for (int i = 0; i < 29; i++) {
    if (!alphabet[i])
      cnt++; // Increments 'cnt' if the character corresponding to index 'i' was not found.
  }

  /**
   * @brief Determines the output message based on the parity of 'cnt'.
   * Block Logic: Checks if the calculated 'cnt' is odd or even.
   * If 'cnt' is odd, prints "CHAT WITH HER!". If 'cnt' is even, prints "IGNORE HIM!".
   *
   * Given that 'cnt' calculates (29 - D) where D is the number of distinct characters present:
   * - If D is odd: (29 - odd) = even. 'cnt' is even. 'cnt % 2' is 0. Output: "IGNORE HIM!".
   * - If D is even: (29 - even) = odd. 'cnt' is odd. 'cnt % 2' is 1. Output: "CHAT WITH HER!".
   *
   * This logic is the inverse of the standard problem statement for "Boy or Girl" (Codeforces 236A),
   * where an odd count of distinct characters implies "CHAT WITH HER!" and an even count implies "IGNORE HIM!".
   * This documentation reflects the observed behavior of the provided code.
   */
  if (cnt % 2) // Checks if 'cnt' is odd.
    cout << "CHAT WITH HER!" << nl; // Output if 'cnt' is odd.
  else
    cout << "IGNORE HIM!" << endl; // Output if 'cnt' is even.

  return 0; // Indicates successful program execution.
}

/**
 * @brief Implementation of FAST_IO function for optimizing C++ I/O.
 */
void FAST_IO() {
  ios_base::sync_with_stdio(0); // Unties C++ streams from C standard streams.
  cin.tie(0);                   // Unties cin from cout.
  cout.tie(0);                  // Unties cout from cin.
// cout.setf(ios::fixed);      // Uncomment for fixed-point notation for floating numbers.
// cout.precision(20);         // Uncomment to set precision for floating numbers.
#ifndef _offline // Conditional compilation for offline debugging.
// freopen("input.txt", "r", stdin);  // Uncomment to redirect standard input from "input.txt".
// freopen("output.txt", "w", stdout); // Uncomment to redirect standard output to "output.txt".
#endif
}
