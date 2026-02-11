/**
 * @file 236A.CPP
 * @brief Determines a user's gender based on the number of distinct characters in their username.
 * @details This program solves the "Boy or Girl" problem on Codeforces. It reads a username,
 * counts the number of unique characters in it, and then outputs "CHAT WITH HER!" if the
 * count is even, or "IGNORE HIM!" if the count is odd.
 *
 * @see https://codeforces.com/problemset/problem/236/A
 *
 * Algorithm:
 * 1. Read the input username into a character array.
 * 2. Sort the character array alphabetically. This groups all identical characters together.
 * 3. Iterate through the sorted array, counting the number of times a character is different
 *    from the one immediately following it. This effectively counts the number of distinct characters.
 * 4. Check if the final count of distinct characters is even or odd.
 * 5. Print the corresponding message.
 *
 * Time Complexity: O(N log N), where N is the length of the username, dominated by the sorting step.
 * Space Complexity: O(N) to store the username string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // str: character array to store the username, size 101 for up to 100 chars + null terminator.
  char str[101];
  // count: stores the number of distinct characters.
  int count = 0, n, i;

  cin >> str;
  n = strlen(str);

  // Sort the string to group identical characters together.
  sort(str, str + n);

  /**
   * @brief Counts the number of distinct characters in the sorted string.
   *
   * Pre-condition: The 'str' array is sorted alphabetically.
   * Logic: By iterating through the sorted array, we can count unique characters.
   * A character is unique each time it differs from the character next to it.
   */
  for (i = 0; str[i]; i++) {
    // The original comment "kata terakhir str adalah '\0'" means "the last character of str is '\0'".
    // This check works because when str[i] is the last occurrence of a character,
    // str[i+1] will either be a different character or the null terminator, causing the condition to be true.
    if (str[i] != str[i + 1]) {
      count++;
    }
  }

  /**
   * @brief Determines the output based on the parity of the distinct character count.
   * If the count is even, the user is assumed to be female.
   * If the count is odd, the user is assumed to be male.
   */
  if (count % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
}
