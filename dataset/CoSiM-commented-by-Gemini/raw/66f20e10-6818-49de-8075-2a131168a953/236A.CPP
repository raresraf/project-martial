#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>
using namespace std;

/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem from Codeforces (Problem 236A).
 * @details The program determines a user's gender for a chat application based on
 * the number of distinct characters in their username. If the count of distinct
 * characters is even, the user is identified as female ("CHAT WITH HER!").
 * If the count is odd, the user is identified as male ("IGNORE HIM!").
 *
 * Algorithm:
 * The program uses a frequency array (acting as a hash set) of size 26 to track
 * the presence of each lowercase English letter. It reads the username, iterates
 * through each character, and marks the corresponding letter as present in the
 * frequency array. After scanning the entire string, it counts the number of
 * marked letters in the array to find the number of distinct characters.
 * Finally, it checks the parity of this count to produce the required output.
 *
 * Time Complexity: O(L), where L is the length of the input string, as the program
 * iterates through the string once.
 * Space Complexity: O(1), because the frequency array has a fixed size (26),
 * independent of the input string length.
 */
int main() {
  char a[100];
  // `b` serves as a frequency map for the 26 lowercase English letters.
  int b[26];
  int flag = 0;
  int i, j, c = 0;

  // Initialize the frequency map to all zeros.
  for (i = 0; i < 26; ++i) {
    b[i] = 0;
  }

  // Read the username string.
  scanf("%s", a);

  /**
   * @brief Populate the frequency map.
   * This loop iterates over the input string. For each character, it calculates
   * its corresponding index (0-25) and increments the count at that index.
   */
  for (i = 0; i < strlen(a); i++) {
    j = a[i];
    j = j - 97; // 'a' has ASCII value 97, so this maps 'a'->0, 'b'->1, etc.
    b[j]++;
  }

  /**
   * @brief Count the number of distinct characters.
   * This loop checks how many letters had a count greater than zero.
   */
  for (i = 0; i < 26; i++) {
    if (b[i] > 0)
      c++;
  }

  /**
   * @brief Check the parity of the distinct character count.
   * If the count `c` is even, the user is female. Otherwise, the user is male.
   */
  if (c % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}