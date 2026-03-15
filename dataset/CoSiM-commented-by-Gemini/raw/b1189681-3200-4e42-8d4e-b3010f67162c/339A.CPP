/**
 * @file 339A.CPP
 * @brief A limited, buggy solution to the "Helpful Maths" problem.
 * @details This program attempts to sort the numbers in a string sum. It uses a
 * counting-sort-like approach that only works if the numbers are '1', '2', or '3'.
 * The logic for reconstructing the output string is complex and contains bugs that
 * can lead to undefined behavior.
 *
 * @algorithm
 * 1. Reads the input string.
 * 2. Iterates through the input string, examining only the digit characters.
 * 3. It segregates the digits by building three new strings: `w1` for '1's, `w2` for '2's,
 *    and `w3` for '3's. Each number is appended with a '+'.
 * 4. It then attempts to print `w1`, `w2`, and `w3` in order, using convoluted logic
 *    to remove the trailing '+' at the correct position.
 *
 * @complexity
 * - Time: O(L), where L is the length of the input string, as it's a single pass.
 * - Space: O(L), as the combined length of `w1`, `w2`, and `w3` is proportional to L.
 *
 * @note The output logic is extremely fragile and bug-prone. For example, if the string `w3`
 * is empty, `w3.size() - 1` results in a very large unsigned number, causing `substr` to have
 * undefined behavior. The code likely passes only a specific subset of test cases.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  string word; // The input string, e.g., "3+1+2".
  // Strings to hold the sorted parts of the sum.
  string w1 = "", w2 = "", w3 = "";
  cin >> word;
  
  // `size` is the number of digits in the sum.
  int size = word.size() / 2 + 1;
  // `temp` is an index to walk through the input string, skipping '+'.
  int temp = 0;

  /**
   * @brief This loop segregates the numbers '1', '2', and '3' into separate strings.
   * This acts as a manual counting sort for a very limited character set.
   */
  while (size--) {
    if (word[temp] == '1')
      w1 += "1+";
    else if (word[temp] == '2')
      w2 += "2+";
    else
      w3 += "3+";

    temp += 2;
  }

  /**
   * @brief Brittle output logic to concatenate the parts and remove trailing '+'.
   * This block of code is highly complex and contains bugs related to string manipulation
   * that can cause incorrect output or undefined behavior, especially with `substr`.
   */
  if (w2.size() > 0 || w3.size() > 0) {
    cout << w1;
    if (w3.size() == 0)
      cout << w2.substr(0, w2.size() - 1);
    else
      cout << w2;
  } else
    cout << w1.substr(0, w1.size() - 1);

  // This line is buggy and can cause a crash if w3 is empty.
  cout << w3.substr(0, w3.size() - 1) << endl;
  return 0;
}