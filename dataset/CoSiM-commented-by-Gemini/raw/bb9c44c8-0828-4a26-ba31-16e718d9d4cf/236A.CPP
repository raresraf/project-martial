/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem.
 *
 * @details This program determines if a user is male or female based on the number of
 * distinct characters in their username. If the count of distinct characters is odd,
 * the user is considered male ("IGNORE HIM!"). If it's even, the user is female ("CHAT WITH HER!").
 *
 * @section
 * Algorithm:
 * The code uses a nested loop approach to count the distinct characters. It initializes a counter
 * with the total length of the string. It then iterates through each character, and for each one,
 * it scans the rest of the string for duplicates. When the first duplicate of a character is found,
 * the counter is decremented.
 *
 * Time Complexity: O(L^2), where L is the length of the username, due to the nested loops.
 * Space Complexity: O(L), to store the username in a character array.
 */
#include <cstring>
#include <ctype.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

int main() {
  char array[105];
  int m;
  cin >> array;
  int n = strlen(array);
  /**
   * @brief This block calculates the number of distinct characters in the input string.
   * Invariant: 'n' is initialized to the total length and is decremented each time
   * a character is found to have a duplicate later in the string.
   */
  for (int i = 0; i < strlen(array); i++) {
    m = 0; // 'm' is a flag to ensure 'n' is decremented only once per character 'i'.
    // This inner loop scans for duplicates of `array[i]`.
    for (int j = i + 1; j < strlen(array); j++) {
      if (array[i] == array[j] && (m == 0)) {
        n--;
        m++;
      }
    }
  }

  /**
   * @brief This block checks the parity of the final count of distinct characters.
   * Logic: If the count 'n' is odd, the user is male. If it's even, the user is female.
   */
  if (n % 2)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
}
