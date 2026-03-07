/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem.
 *
 * @details
 * This program processes a list of words. For each word, it checks if its
 * length is greater than 10. If it is, the word is replaced by a special
 * abbreviation. Otherwise, the word is printed as is.
 *
 * Algorithm:
 * 1. Read the number of words, n.
 * 2. Loop n times:
 *    a. Read a word.
 *    b. Get its length.
 *    c. If the length is greater than 10, print the first character,
 *       the length minus 2, and the last character.
 *    d. Otherwise, print the original word.
 *
 * Example:
 * "localization" -> "l10n"
 * "word" -> "word"
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum length of a word, due to I/O operations. The processing for each word is O(1).
 * Space Complexity: O(L) to store the longest word.
 */
#include <bits/stdc++.h>
using namespace std;

// Typedefs for common data types, often used in competitive programming for brevity.
typedef long long ll;
typedef pair<int, int> ii;

int main() {
  /**
   * Functional Utility: These lines optimize C++ I/O operations.
   * `ios_base::sync_with_stdio(0)` disables synchronization with C-style I/O.
   * `cin.tie(0)` and `cout.tie(0)` untie cin and cout from each other.
   * This is a standard optimization in competitive programming for faster execution.
   */
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  int n;
  cin >> n; // Read the number of words to process.
  
  /**
   * @brief Main processing loop that iterates through each word.
   * Pre-condition: 'n' contains the total number of words.
   * Invariant: 'i' words have been processed.
   */
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    int sz = s.size();
    /**
     * @brief Conditional block to check word length.
     * If the word is not "too long" (length <= 10), it is printed as is
     * and the loop continues to the next iteration.
     */
    if (sz <= 10) {
      cout << s << endl;
      continue;
    }
    // If the word is "too long", print its abbreviation:
    // first character, count of middle characters, and last character.
    cout << s.front() << sz - 2 << s.back() << endl;
  }
  return 0;
}
