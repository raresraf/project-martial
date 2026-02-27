/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem from Codeforces.
 * @details The problem is to determine if a given string contains "hello" as
 * a subsequence. A subsequence is a sequence of characters that appear in the
 * same order in the string, but not necessarily consecutively.
 *
 * @algorithm
 * The algorithm uses a two-pointer approach to find the subsequence.
 * 1. Define the target subsequence, "hello".
 * 2. Read the input string.
 * 3. Use one pointer `i` to iterate through the input string and another
 *    pointer `j` to iterate through the target subsequence "hello".
 * 4. Traverse the input string with `i`. If the character at `word[i]` matches
 *    the character at `etalon[j]`, it means we have found the next character
 *    of the subsequence, so we advance the pointer `j`.
 * 5. Regardless of whether a match is found, `i` always advances.
 * 6. After the loop, if the pointer `j` has reached the end of the target
 *    subsequence "hello", it means all its characters were found in the correct
 *    order.
 * 7. Print "YES" if the subsequence was found, otherwise print "NO".
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string, because
 * we iterate through it only once.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // The target subsequence. "etalon" is French for "standard".
  string etalon = "hello";
  string word; // The input string to be checked.
  cin >> word;

  // Pointer 'j' for the 'etalon' string.
  size_t j = 0;
  
  /**
   * @block
   * @brief Iterate through the input string to find the "hello" subsequence.
   * @invariant Pointer 'i' traverses the input 'word'. Pointer 'j' only advances
   * when a character of the 'etalon' subsequence is found in order.
   */
  for (size_t i = 0; i < word.size() && j < etalon.size(); ++i) {
    if (word[i] == etalon[j])
      ++j; // Found the next character of the subsequence, advance 'j'.
  }

  /**
   * @brief Check if the entire subsequence was found.
   * If 'j' equals the length of "hello", it means every character was
   * found in the correct order.
   */
  string answer = ((etalon.size() == j) ? "YES" : "NO");
  cout << answer << endl;
  
  return 0;
}
