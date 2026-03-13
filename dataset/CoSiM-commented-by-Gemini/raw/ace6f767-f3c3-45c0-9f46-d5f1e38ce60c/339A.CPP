/**
 * @file 339A.CPP
 * @brief Rearranges a sum of numbers into non-decreasing order.
 *
 * This program takes a string representing a sum of numbers (containing only
 * 1, 2, and 3) and reorders the numbers in non-decreasing order, separated by
 * '+'.
 *
 * @see https://codeforces.com/problemset/problem/339/A
 */
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  string s;
  cin >> s;
  string ans;
  vector<int> arr;
  if (s.length() == 1) {
    cout << s;
  } else {
    /**
     * @brief Parses the input string and extracts the numbers.
     *
     * This loop iterates through the input string `s`. It ignores the '+'
     * characters and pushes the integer value of the other characters
     * into a vector `arr`.
     */
    for (int i = 0; i < s.length(); i++) {
      if (s[i] == '+')
        continue;
      if (s[i] == '1') {
        arr.push_back(1);
      } else if (s[i] == '2') {
        arr.push_back(2);
      } else {
        arr.push_back(3);
      }
    }
    // Sorts the vector of numbers in non-decreasing order.
    sort(arr.begin(), arr.end());
    vector<int>::iterator ele;

    /**
     * @brief Constructs and prints the rearranged sum.
     *
     * This loop iterates through the sorted vector `arr` and prints each
     * number, followed by a '+' sign, except for the last number.
     */
    for (ele = arr.begin(); ele != arr.end(); ele++) {
      cout << *ele;
      if (ele != arr.end() - 1)
        cout << "+";
    }
  }

  return 0;
}
