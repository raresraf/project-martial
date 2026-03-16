/**
 * @file 236A.CPP
 * @brief An attempted solution to the "Boy or Girl" problem (Codeforces 236A).
 *
 * @details
 * This program determines if a user is male or female based on the number of
 * distinct characters in their username. The problem states that if the number of
 * distinct characters is even, the user is female ("CHAT WITH HER!"), and if it's
 * odd, the user is male ("IGNORE HIM!").
 *
 * The code correctly uses a `std::set` to count the number of distinct characters
 * in the input username. However, it incorrectly implements the final step: the
 * logic for printing the output message is inverted. It prints the message for a
 * male when the user is female, and vice-versa.
 *
 * @warning The output logic is flawed. The messages for even and odd counts are
 * swapped compared to the problem description.
 *
 * Algorithm:
 * 1. Read a username string.
 * 2. Use a `std::set` to find the number of unique characters in the string.
 * 3. Check if the count of unique characters is even or odd.
 * 4. Print the result based on the parity (with inverted logic).
 *
 * Time Complexity: O(L), where L is the length of the username string, as each
 *                  character is inserted into the set once.
 * Space Complexity: O(K), where K is the number of distinct characters (at most 26
 *                   for the English alphabet).
 */
#include <cstring>
#include <iostream>
#include <set>
using namespace std;
int main() {
  set<char> a;
  char b[101];
  int i;

  // Reads a line from stdin, including the trailing newline, into buffer `b`.
  fgets(b, sizeof(b), stdin);

  /**
   * @brief Populates a set with characters from the input string to find unique ones.
   * @details The `std::set` automatically handles duplicates, so after this loop,
   * `a.size()` will hold the count of distinct characters. Note that this
   * count will include the newline character `\n` read by `fgets`.
   */
  for (i = 0; i < strlen(b); i++) {
    a.insert(b[i]);
  }

  i = a.size();

  /**
   * @brief Checks the parity of the distinct character count to determine gender.
   * @warning This conditional block implements the logic incorrectly. It prints
   * "IGNORE HIM!" for an even count and "CHAT WITH HER!" for an odd count, which
   * is the reverse of the problem's requirements.
   */
  if (i % 2 == 0)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
  return 0;
}
