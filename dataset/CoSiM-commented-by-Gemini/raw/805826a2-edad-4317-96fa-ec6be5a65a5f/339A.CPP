/**
 * @file 339A.CPP
 * @brief Solution to a competitive programming problem involving reordering a sum of numbers.
 * @details The program is intended to read a string representing a sum of single-digit numbers
 * (e.g., "3+2+1"), sort the numbers, and print the new sum (e.g., "1+2+3").
 *
 * @algorithm The intended algorithm is to extract the numbers from the string, sort them,
 * and then reconstruct the string with the sorted numbers.
 * @complexity
 * Time complexity: Intended to be O(N log N) due to sorting, where N is the number of digits.
 * Space complexity: O(N) to store the numbers in a vector.
 * @note The current implementation has logical errors, including infinite loops.
 */
#include <bits/stdc++.h>

using namespace std;

void convert(string &c, int s);

/**
 * @brief Main function to read the input and call the conversion function.
 */
int main() {
  string cadena = {};

  while (cin >> cadena) {
    if (cadena.size() > 1) {
      convert(cadena, cadena.size());
    } else
      cout << cadena << endl;
  }

  return 0;
}

/**
 * @brief Attempts to sort the numbers in the input string.
 * @param c The input string representing the sum.
 * @param s The size of the string.
 * @note This function contains infinite loops as the loop increment `i + 2` does not modify `i`.
 */
void convert(string &c, int s) {
  vector<int> sumas = {};
  int num_sig = 0, num_ele = 0, j = 0;

  // The logic inside this if-else block is flawed and leads to infinite loops.
  if ((s % 2) == 0) { // si es par, el n mero de "+" es impar. tam/2 +1
    // num_sig = (s/2)+1;
    // num_ele = s - num_sig;
    /**
     * @brief Intended to extract the numbers from the string.
     * @note This is an infinite loop because 'i' is not incremented.
     */
    for (int i = 0; i < s; i + 2) {
      sumas.push_back(c[i]);
    }

    sort(sumas.begin(), sumas.end());

    /**
     * @brief Intended to print the sorted numbers with '+' signs.
     * @note This is an infinite loop because 'i' is not incremented.
     */
    for (int i = 0; i < s; i + 2) {
      cout << sumas[++j];
      if (i != s - 1)
        cout << "+";
    }
    j = 0;
  } else { // si es impar, el n mero de "+" es par. tam/2 +1
    /**
     * @brief Intended to extract the numbers from the string.
     * @note This is an infinite loop because 'i' is not incremented.
     */
    for (int i = 0; i < s; i + 2) {
      sumas.push_back(c[i]);
    }

    sort(sumas.begin(), sumas.end());

    /**
     * @brief Intended to print the sorted numbers with '+' signs.
     * @note This is an infinite loop because 'i' is not incremented.
     */
    for (int i = 0; i < s; i + 2) {
      cout << sumas[++j];
      if (i != s - 1)
        cout << "+";
    }
    j = 0;
  }
  cout << endl;
}