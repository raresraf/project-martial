/**
 * @file 58A.CPP
 * @brief Solution to the "Chat room" problem (Codeforces 58A).
 *
 * Problem Statement:
 * Given a string, determine if it is possible to form the word "hello" by deleting
 * zero or more characters from it. This is equivalent to checking if "hello" is
 * a subsequence of the given string.
 *
 * Algorithm:
 * The program uses a two-pointer approach to solve this subsequence problem
 * efficiently. One pointer, 'i', iterates through the input string from beginning
 * to end. A second pointer, 'j', tracks the current character we are looking for
 * in the target word "hello".
 *
 * As the code scans the input string with 'i', it compares the current character
 * `str[i]` with the target character `str2[j]`. If they match, it means we have
 * found the next character of the "hello" subsequence, so we advance the target
 * pointer 'j'. This process continues until either 'j' has advanced past the
 * end of "hello" (meaning all characters were found in order), or the input
 * string is fully scanned.
 *
 * Time Complexity: O(L), where L is the length of the input string, as it
 * only requires a single pass.
 * Space Complexity: O(L_max), for the character array used to store the input.
 */
#include <cstring>
#include <iostream>

int main() {
  int i, j, flag = 0;
  char str[103];
  char str2[5] = {'h', 'e', 'l', 'l', 'o'};
  std::cin >> str;

  /**
   * Block Logic: Use a two-pointer approach to find the "hello" subsequence.
   * 'i' iterates through the input string `str`.
   * 'j' tracks the position in the target string `str2` ("hello").
   */
  for (i = 0, j = 0; i < strlen(str); i++) {
    // If the current character in the input matches the target character...
    if (str[i] == str2[j]) {
      // ...move to the next character in the target.
      j++;
    }
    // If we have found all 5 characters of "hello"...
    if (j == 5) {
      // ...set the flag and exit the loop.
      flag = 1;
      break;
    }
  }

  // Check the flag to determine the output.
  if (flag == 1)
    std::cout << "YES" << '
';
  else if (flag == 0)
    std::cout << "NO" << '
';
  return 0;
}
