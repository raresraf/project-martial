#include <bits/stdc++.h>
using namespace std;

/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem from Codeforces (Problem 71A).
 * @details This program abbreviates words that are longer than 10 characters. The
 * abbreviation consists of the first character, the number of characters between the
 * first and last, and the last character.
 *
 * Algorithm:
 * The program reads the number of words to process, `n`. It then loops `n` times.
 * In each iteration, it reads a word. If the word's length is greater than 10,
 * it modifies the string in-place. The new string is constructed from the original's
 * first character, followed by the string representation of its length minus two,
 * and finally the original's last character. If the word is not longer than 10
 * characters, it remains unchanged. The resulting string is then printed.
 *
 * Time Complexity: O(n * L), where `n` is the number of words and L is the average
 * length of a word. String construction and I/O are the main operations.
 * Space Complexity: O(L_max), where L_max is the maximum length of a word, to store
 * the string.
 */
int main() {
  int n;
  string in;
  // Read the number of words.
  scanf("%d", &n);
  while (n--) {
    cin >> in;
    // Check if the word needs to be abbreviated.
    if (in.size() > 10) {
      // Calculate the number of characters between the first and last.
      int len = in.size() - 2;
      // Reconstruct the string in the abbreviated format.
      // `to_string(len)` converts the integer length to its string representation.
      in = in[0] + to_string(len) + in.back();
    }
    cout << in << endl;
  }
}
