/**
 * @file raw/44eb4107-de42-4160-9899-8bfeadd84be9/236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl.
 *
 * @algorithm
 * The problem requires determining if the number of distinct characters in a
 * username is even or odd to classify a user as female ("CHAT WITH HER!") or
 * male ("IGNORE HIM!").
 *
 * This solution uses a frequency array to count the occurrences of each
 * character, which is a classic and efficient method for character counting
 * problems with a small, fixed alphabet (like lowercase English letters).
 *
 * The algorithm is as follows:
 * 1. Read the username into a `std::string`.
 * 2. Create an integer array `a` of size 26, initialized to all zeros. This
 *    array will serve as a frequency map for the letters 'a' through 'z'.
 * 3. Iterate through each character of the input string. For each character,
 *    calculate its 0-based alphabet index (e.g., by subtracting the ASCII
 *    value of 'a', which is 97) and increment the count at that index in
 *    the frequency array `a`.
 * 4. After counting, iterate through the frequency array `a`. Count how many
 *    elements are greater than 0. This count, `res`, represents the number
 *    of distinct characters in the original string.
 * 5. Check the parity of `res`. If `res` is even, print "CHAT WITH HER!".
 *    Otherwise, print "IGNORE HIM!".
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the string. The first loop
 *   iterates L times, and the second loop runs a constant 26 times.
 * - Space Complexity: O(1), as the frequency array `a` has a fixed size (26),
 *   which is constant and does not depend on the input string length.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  
  // `a` is a frequency array for the 26 lowercase English letters.
  int a[26] = {0};

  /**
   * @brief Populate the frequency array.
   * For each character in the string, increment the corresponding counter.
   */
  for (int i = 0; i < s.length(); i++) {
    // s[i] - 97 converts a character ('a'-'z') to an index (0-25).
    // 97 is the ASCII value of 'a'.
    a[s[i] - 97]++;
  }
  
  // `res` will store the count of distinct characters.
  int res = 0;

  /**
   * @brief Count the number of distinct characters from the frequency array.
   */
  for (int i = 0; i < 26; i++) {
    // If the frequency of a character is greater than 0, it was present.
    if (a[i] > 0)
      res++;
  }

  /**
   * @brief Check the parity of the distinct character count.
   */
  if (res % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}
