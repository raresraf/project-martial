/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces (Problem 71A).
 * This program abbreviates words longer than 10 characters by replacing the middle
 * part with the count of characters between the first and last letters.
 *
 * Algorithm: Conditional abbreviation based on string length.
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum length of a word.
 *                  String length check and character access are O(1) or O(L) depending on implementation.
 * Space Complexity: O(L) for storing the current word.
 */

#include <bits/stdc++.h> // @brief Includes many standard libraries (e.g., iostream, string, algorithm).

using namespace std;

typedef long long ll;         // @brief Type definition for long long integers.
typedef pair<int, int> ii; // @brief Type definition for a pair of integers.

int main() {
  /**
   * @brief Main function to process words according to the "Way Too Long Words" rules.
   * Reads a number of words and for each, prints either the original word or its
   * abbreviated form if it exceeds a certain length.
   */
  // Functional Utility: Unties cin/cout from C standard streams and disables synchronization
  // with C stdio, which can significantly speed up I/O operations in competitive programming.
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  
  int n; // @brief Number of words to process.
  cin >> n;
  
  // Block Logic: Iterate through each word, apply abbreviation rules, and print the result.
  for (int i = 0; i < n; i++) {
    string s; // @brief The current word being processed.
    cin >> s;
    int sz = s.size(); // @brief Length of the current word.
    
    // Block Logic: Conditional abbreviation based on word length.
    // If word length is less than or equal to 10, print the word as is.
    if (sz <= 10) {
      cout << s << endl;
      continue; // Move to the next word.
    }
    // Otherwise, print the abbreviated form: first char + (length - 2) + last char.
    cout << s.front() << sz - 2 << s.back() << endl;
  }
  return 0; // @brief Indicates successful program termination.
}
