/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem from Codeforces.
 * @details The problem states that a user's gender is determined by the number
 * of distinct characters in their username. If the count of distinct characters
 * is even, the user is female ("CHAT WITH HER!"). If it's odd, the user is male
 * ("IGNORE HIM!").
 *
 * @algorithm
 * A frequency array (or a set) is used to count the number of distinct
 * characters in the input username.
 * 1. Read the username string.
 * 2. Create an integer array `arr` of size 26, initialized to zeros. This
 *    array will act as a boolean map for the English alphabet.
 * 3. Iterate through each character of the username.
 *    a. For each character, calculate its 0-based alphabet index (e.g., 'a' -> 0).
 *       The code does this by subtracting the ASCII value of 'a' (97).
 *    b. Mark the presence of this character by setting the value at that index
 *       in `arr` to 1.
 * 4. After the first loop, iterate through the frequency array `arr` from 0 to 25.
 * 5. Count how many elements in `arr` are 1. This gives the total number of
 *    distinct characters.
 * 6. Check if the final count is even or odd and print the corresponding message.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the username, as the process
 * involves a single pass over the string. The second loop is constant time (26 iterations).
 * Space Complexity: O(1), as the frequency array has a fixed size (26).
 */
// 236A
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  
  // A frequency array to mark the presence of each letter of the alphabet.
  int arr[27] = {0};
  int count = 0; // To store the count of distinct characters.

  /**
   * @block
   * @brief Mark the presence of each character in the frequency array.
   */
  for (int i = 0; i < s.length(); i++) {
    // `s[i] - 97` converts a lowercase char to a 0-25 index. ('a' is ASCII 97)
    arr[s[i] - 97] = 1;
  }

  /**
   * @block
   * @brief Count the number of distinct characters found.
   */
  for (int i = 0; i < 26; i++) {
    if (arr[i] == 1)
      count++;
  }

  /**
   * @brief Determine the output based on the parity of the distinct character count.
   */
  if (count % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
    
  return 0;
}
