/**
 * @file 339A.CPP
 * @brief Solution attempt for Codeforces problem 339A - Helpful Maths.
 *
 * @algorithm
 * The problem requires sorting the numbers in a string representing a sum
 * (e.g., "3+1+2" should become "1+2+3").
 *
 * The intended algorithm here is a selection sort directly on the input string.
 * It considers only the characters at even indices (0, 2, 4, ...), which are
 * the numbers, and attempts to sort them in place.
 *
 * @warning This implementation has a significant logical flaw. It prints each
 * number immediately after finding its supposed final position within the outer
 * loop of the selection sort. The sorting process is not complete when printing
 * begins. As a result, the output will be incorrect for most inputs because
 * the sorting is interleaved with printing. The correct approach would be to
 * complete the sort fully and then iterate through the modified string to print
 * the result.
 *
 * @note The line `#define long long ll;` is a syntax error and has no effect.
 *
 * @complexity
 * - Intended Time Complexity: O(N^2), where N is the number of summands, due to
 *   the nested loops implementing a selection sort.
 * - Space Complexity: O(L), where L is the length of the input string.
 */
#include <iostream>

using namespace std;

#define long long ll; // This is a syntax error and is unused.

string s;

void solve() {
  string answer;
  /**
   * @brief This nested loop attempts to implement a selection sort on the numbers
   * (at even indices) in the string.
   *
   * FLAW: The outer loop prints the character at `s[i]` before the entire
   * sort is complete. It should sort first, then print in a separate loop.
   */
  for (int i = 0; i < s.length(); i = i + 2) {
    // Inner loop to find the minimum element in the rest of the string.
    for (int j = i; j < s.length(); j = j + 2) {
      if (s[i] > s[j]) {
        // Swap to bring the minimum element to position `i`.
        int aux = s[i];
        s[i] = s[j];
        s[j] = aux;
      }
    }
    // This print statement is executed prematurely.
    if (i != s.length() - 1)
      cout << s[i] << "+";
    else
      cout << s[i];
  }
}

int main() {
  cin >> s;

  solve();
}