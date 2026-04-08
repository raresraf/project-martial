/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem from Codeforces (Problem 236A).
 * @details The program determines whether to chat with a user based on the number
 * of distinct characters in their username. If the count of distinct characters
 * is even, the output is "CHAT WITH HER!". If it's odd, the output is "IGNORE HIM!".
 *
 * @algorithm
 * 1. Read the user's name as a string.
 * 2. Create a frequency map (an array of size 26) to store the count of each
 *    lowercase English letter.
 * 3. Iterate through the input string and update the frequency for each character.
 * 4. Count the number of characters that appeared at least once (i.e., the number
 *    of distinct characters).
 * 5. Check if the count of distinct characters is even or odd.
 * 6. Print the corresponding message.
 *
 * @time_complexity O(N), where N is the length of the username string.
 * @space_complexity O(1), as the frequency array size is constant (26).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Frequency array for lowercase English letters 'a' through 'z'.
  int num[26], i;
  for (i = 0; i < 26; i++) {
    num[i] = 0;
  }

  char arr[101];
  cin.getline(arr, 101);

  int co = 0; // Counter for distinct characters.
  int x;
  int len = strlen(arr);

  /**
   * @block_logic Populate the frequency map based on the input string.
   * @pre `arr` contains the username.
   * @post `num` array stores the frequency of each character.
   */
  for (i = 0; i < len; i++) {
    x = arr[i] - 'a';
    num[x]++;
  }

  /**
   * @block_logic Count the number of distinct characters.
   * @pre `num` array is populated.
   * @post `co` holds the count of unique characters in the username.
   */
  for (i = 0; i < 26; i++) {
    if (num[i] > 0) {
      co++;
    }
  }

  /**
   * @block_logic Determine the output based on the parity of the distinct character count.
   */
  if (co % 2 == 0) {
    printf("CHAT WITH HER!\n");
  } else {
    printf("IGNORE HIM!\n");
  }
}
