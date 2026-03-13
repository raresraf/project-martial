/**
 * @file 339A.CPP
 * @brief Solution for a string problem, likely CodeForces 1154A "Restoring Three Numbers",
 * but the code seems to solve a different problem about forming the word "nineteen".
 *
 * @note The filename `339A.CPP` (Helpful Maths) is incorrect for the logic implemented.
 * The logic appears to solve the problem of finding the maximum number of times the
 * word "nineteen" can be formed from the letters of a given input string.
 *
 * @details
 * To form the word "nineteen", one needs:
 * - 3 'n's
 * - 1 'i'
 * - 3 'e's
 * - 1 't'
 *
 * @algorithm
 * 1. Count the frequency of the characters 'n', 'i', 'e', 't' in the input string.
 * 2. Calculate how many sets of required letters can be formed:
 *    - For 'n', we need 3. The formula `(count('n') - 1) / 2` calculates how many pairs of 'n's
 *      are available after setting one aside. This is the limiting factor for the three 'n's.
 *    - For 'i', we need 1, so the number of sets is just the count of 'i'.
 *    - For 'e', we need 3, so the number of sets is `count('e') / 3`.
 *    - For 't', we need 1, so the number of sets is the count of 't'.
 * 3. The overall maximum number of "nineteen"s is the minimum of these calculated set counts.
 *
 * @time_complexity O(L), where L is the length of the input string.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  // Counters for the required letters. 'n_count' is named 'n' in the code.
  int i_count = 0, n_count = 0, t_count = 0, e_count = 0;
  cin >> str;

  /**
   * @brief Count the frequency of required characters.
   */
  for (int j = 0; j < str.size(); j++) {
    if (str[j] == 'n')
      n_count++;
    else if (str[j] == 't')
      t_count++;
    else if (str[j] == 'e')
      e_count++;
    else if (str[j] == 'i')
      i_count++;
  }
  
  /**
   * @brief Calculate the number of possible sets for each character.
   * This block determines the bottleneck character.
   */

  // For the letter 'n', we need three. The logic `(n_count - 1) / 2` determines
  // how many times we can form the word, as we need one 'n' plus two more.
  // This is only valid if n_count >= 1.
  if (n_count > 0) {
    n_count = (n_count - 1) / 2;
  } else {
    n_count = 0;
  }
  
  // For the letter 'e', we need three.
  e_count /= 3;

  // The number of 'i's and 't's are taken as is, since we only need one of each.
  
  // Find the minimum of all the possible sets, which is the answer.
  cout << min({n_count, e_count, i_count, t_count});
  
  return 0;
}
