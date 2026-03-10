/**
 * @file 71A.CPP
 * @brief This program solves the "Way Too Long Words" problem (Codeforces 71A).
 *   It abbreviates words longer than 10 characters to their first letter,
 *   the count of intermediate characters, and their last letter. Shorter words are printed as is.
 *   Problem description: https://codeforces.com/problemset/problem/71/A
 * * Algorithm: String length check and conditional formatting.
 *   For words longer than 10 characters, it calculates the number of characters
 *   between the first and last (which is `length - 2`) and formats the word.
 * Time Complexity: O(N * L) where N is the number of words and L is the average length of a word.
 * Space Complexity: O(L_max) to store the longest input word.
 */
#include <algorithm> // Included but not directly used in the core logic.
#include <cmath>     // Included but not directly used.
#include <cstring>   // Included but not directly used.
#include <iostream>  // Required for std::cin, std::cout, std::endl.
#include <map>       // Included but not directly used.
#include <set>       // Included but not directly used.
#include <utility>   // Included for std::pair, but not directly used.
#include <vector>    // Included but not directly used.

#define MX 100000 // Constant for maximum size.
#define fs first  // Functional Utility: Alias for the first element of a pair.
#define sec second // Functional Utility: Alias for the second element of a pair.
#define TC                                                                     \
  int T;                                                                       \
  cin >> T;                                                                    \
  while (T--) // Functional Utility: Macro for handling multiple test cases.
#define loop(i, a, c) for (int i = 0; i < a; i++) // Functional Utility: Macro for a common for-loop pattern.
#define loop1(i, a, c)                                                         \
  for (int i = 1; i <= a; i++) // Functional Utility: Macro for a 1-indexed for-loop pattern.
#define pb(a) push_back(a)     // Functional Utility: Alias for push_back method of vectors.
#define all(a) (a).begin(), (a).end() // Functional Utility: Macro for getting iterators to the beginning and end of a container.
#define mp(x, y) make_pair((x), (y)) // Functional Utility: Macro for std::make_pair.
#define ll long long int           // Functional Utility: Alias for long long int.
using namespace std;

/**
 * @brief Main function to process words and apply abbreviation rules.
 * Functional Utility: Reads the number of words, then processes each word
 *                     according to its length, applying abbreviation if necessary.
 */
int main() {
  int t; // Stores the total number of test cases (words).
  // Block Logic: Reads the total number of test cases `t`.
  cin >> t;
  /**
   * Block Logic: Iterates `t` times to process each word.
   * Pre-condition: `t` is a non-negative integer.
   * Invariant: Each iteration processes one word. `t--` decrements the counter.
   */
  while (t--) {
    string s; // Stores the current word being processed.
    // Block Logic: Reads the current word.
    cin >> s;
    /**
     * Conditional Logic: Checks if the current word's length is 10 or less.
     * If true, the word is printed as is.
     */
    if (s.length() <= 10) {
      cout << s << endl;
    }
    /**
     * Conditional Logic: If the word's length is greater than 10, it is abbreviated.
     * The abbreviation consists of the first character, the count of intermediate characters
     * (which is `s.length() - 2`), and the last character.
     * Inline: `s[0]` accesses the first character.
     * Inline: `s.length() - 2` calculates the count of intermediate characters.
     * Inline: `s[s.length() - 1]` accesses the last character.
     */
    else {
      cout << s[0];                     // Prints the first character.
      cout << s.length() - 2;           // Prints the count of intermediate characters.
      cout << s[s.length() - 1] << endl; // Prints the last character and a newline.
    }
  }
}
