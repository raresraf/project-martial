/**
 * @file 58A.CPP
 * @brief A. Chat room - Subsequence check for the word "hello".
 * @author Ikki
 * @date Jun 10, 2015
 *
 * This program determines if the word "hello" is a subsequence of an input string.
 * It reads a string from standard input and checks if the characters 'h', 'e', 'l', 'l', 'o'
 * appear in that order within the input string, although not necessarily contiguously.
 *
 * @algorithm Subsequence Matching using a Two-Pointer Approach
 * 1. The target word is "hello".
 * 2. Read the input string `str`.
 * 3. Use two pointers (indices): `i` for the input string `str` and `j` for the target word "hello".
 * 4. Iterate through the input string `str` with pointer `i`.
 * 5. If the character `str[i]` matches the character `hello[j]`, it means we've found the next
 *    character of our target subsequence. Increment `j` to look for the next character of "hello".
 * 6. The loop continues until the end of the input string is reached (`i` loop) or until all
 *    characters of "hello" have been found (`j < 5` condition).
 * 7. After the loop, if `j` is equal to 5, it signifies that all five characters of "hello" were
 *    found in the correct order. The program prints "YES".
 * 8. Otherwise, it prints "NO".
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the input string `str`, as we iterate through it once.
 * Space Complexity: O(N) to store the input string `str`.
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  // `str` will hold the input string from the user.
  string str;
  cin >> str;
  
  // `hello` is the target subsequence we are searching for.
  string hello = "hello";
  
  // `j` is an index for the `hello` string. It counts how many characters of "hello" we have successfully matched.
  int j = 0;
  
  /**
   * @brief This loop iterates through the input string `str`.
   * `i` is the index for `str`.
   * The loop terminates if we either exhaust the input string (`i < str.size()`) or find all characters of "hello" (`j < 5`).
   */
  for (int i = 0; i < (int)str.size() && j < 5; i++)
    // Invariant: `j` holds the number of matched characters from the start of "hello".
    // If the current character in `str` matches the character we're looking for in `hello`.
    if (str[i] == hello[j])
      // Move to the next character in the `hello` string.
      j++;
      
  // After the loop, check if all 5 characters of "hello" were found.
  // The ternary operator `(condition ? value_if_true : value_if_false)` is used for a concise if-else.
  cout << (j == 5 ? "YES" : "NO") << endl;
  
  return 0;
}
