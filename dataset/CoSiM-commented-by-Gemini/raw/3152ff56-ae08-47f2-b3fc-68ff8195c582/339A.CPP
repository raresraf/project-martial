/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem by rearranging a sum.
 * @details This program takes a string representing a sum of single-digit numbers
 * (e.g., "3+1+2") and rearranges it so that the numbers are in non-decreasing
 * order (e.g., "1+2+3").
 *
 * @algorithm The core logic involves three steps:
 * 1. Parsing: The input string is traversed to extract only the numeric characters,
 *    which are stored in a `std::vector`.
 * 2. Sorting: The vector of numbers is sorted in non-decreasing order using `std::sort`.
 * 3. Reconstruction: The sorted sum string is rebuilt by printing the numbers from the
 *    sorted vector, separated by '+' signs.
 *
 * @time_complexity O(N log N), where N is the number of summands. The sorting step is the
 * dominant factor in the complexity.
 * @space_complexity O(N), to store the extracted numbers in a vector.
 */
// link : https://codeforces.com/problemset/problem/339/A

#include <bits/stdc++.h>

using namespace std;

string s;

vector<char> st;

main() {

  cin >> s;

  /**
   * Block Logic: Parse the input string to extract the numbers.
   * The loop iterates through the string, and any character that is not a '+'
   * is considered a number and added to the `st` vector.
   */
  for (int i = 0; i < s.size(); i++) {

    if (s[i] != '+')
      st.push_back(s[i]);
  }

  // Sort the vector of number characters in non-decreasing order.
  sort(st.begin(), st.end());

  /**
   * Block Logic: Reconstruct and print the sorted sum string.
   * Invariant: At each iteration, the smallest remaining number is printed.
   * The loop continues until all numbers have been printed.
   */
  while (!st.empty()) {

    cout << *st.begin();

    st.erase(st.begin());

    // Append a '+' separator as long as there are more numbers to print.
    if (!st.empty())
      cout << "+";
  }
}
