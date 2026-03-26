/**
 * @file 266A.CPP
 * @brief This program counts the number of adjacent identical characters in a given string.
 * It reads a string and iterates through it, comparing each character with its successor.
 * Algorithm: Linear scan with adjacency check.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

/**
 * @brief Main function to count adjacent identical characters in an input string.
 * It reads an integer 'n' (though unused in the current logic), then a string,
 * and finally prints the count of adjacent identical characters.
 * @return Implicitly returns 0 on successful execution.
 */
int main() {
  // @var n: Integer, potentially intended for string length or number of test cases, but unused in this implementation.
  // @var i: Loop counter variable.
  // @var co: Integer counter to store the number of adjacent identical characters found. Initialized to 0.
  int n, i, co = 0;
  // Read an integer 'n' from standard input. In this specific problem, 'n' often represents the string length but is not used directly to control array bounds or loop iterations for 'arr'.
  scanf("%d", &n);
  // Consume the newline character left in the buffer by scanf, to prevent cin.getline from reading it as an empty line.
  getchar();
  // @var arr: Character array to store the input string. Maximum size 51 (for up to 50 characters + null terminator).
  char arr[51];
  // Read a line of text into 'arr', up to 50 characters plus the null terminator.
  cin.getline(arr, 51);
  // @var len: Integer to store the actual length of the string read into 'arr'.
  int len = strlen(arr);
  /**
   * @brief Loop through the string to compare adjacent characters.
   * The loop runs up to 'len - 1' because it compares `arr[i]` with `arr[i + 1]`.
   * Invariant: 'co' accurately reflects the count of adjacent identical characters
   * from the beginning of the string up to the character at `arr[i]`.
   */
  for (i = 0; i < len - 1; i++) {
    // Check if the current character is identical to the next character.
    if (arr[i] == arr[i + 1]) {
      // If they are identical, increment the counter 'co'.
      co++;
    }
  }
  // Print the final count of adjacent identical characters followed by a newline.
  printf("%d\n", co);
}
