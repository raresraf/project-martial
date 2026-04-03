/**
 * @file 339A_2.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces.
 * @see https://codeforces.com/problemset/problem/339/A
 *
 * @description
 * This program takes a string representing a sum of single-digit numbers (1, 2, or 3),
 * such as "3+2+1", and reorders the numbers in non-decreasing order to produce a new sum,
 * for example, "1+2+3".
 *
 * @algorithm
 * 1. Read the entire input string which represents the sum.
 * 2. Manually parse the string to extract the individual numbers (addends). The numbers are separated by '+'.
 * 3. Store these numbers in an integer vector.
 * 4. Sort the vector of numbers in non-decreasing order.
 * 5. Reconstruct the output string by concatenating the sorted numbers, separated by '+'.
 * 6. Print the final reconstructed string.
 *
 * @time_complexity O(L + K log K), where L is the length of the input string and K is the number of digits.
 *                  The parsing is O(L) and the sorting is O(K log K). Since K is proportional to L, this is dominated by sorting.
 * @space_complexity O(K) to store the extracted numbers in a vector.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  vector<int> vec;
  string s;
  string tmp;
  cin >> s;
  int len = s.size();
  int left = 0;

  /**
   * @brief Block Logic: Parsing the input string to extract numbers.
   * This loop iterates through the string, building up number substrings. When a '+' is encountered,
   * the accumulated number string is converted to an integer and pushed into a vector.
   * This handles all numbers except the very last one.
   */
  while (left < len) {
    if (s[left] != '+') {
      char p[2] = {s[left], 0};
      string t = p;
      tmp += t;
      left++;
    } else {
      int tt = stoi(tmp);
      vec.push_back(tt);
      tmp = "";
      left++;
    }
  }

  /**
   * @brief Block Logic: Handling the last number in the sum.
   * The main parsing loop doesn't add the final number to the vector because there's no trailing '+'.
   * This block extracts the last character, converts it to an integer, and adds it to the vector.
   */
  char t1[2] = {s[s.size() - 1], 0};
  string sss = t1;

  int ttt = stoi(sss);
  vec.push_back(ttt);
  string ret = "";
  
  // Sort the numbers in non-decreasing order.
  sort(vec.begin(), vec.end());
  
  /**
   * @brief Block Logic: Reconstructing the output string.
   * Iterates through the sorted vector of numbers and builds the final output string,
   * with each number separated by a '+'.
   */
  for (auto i : vec) {
    ret += to_string(i) + '+';
  }
  
  // Remove the trailing '+' from the end of the string.
  ret = ret.substr(0, ret.size() - 1);
  
  // Output the final, sorted sum.
  cout << ret;
  return 0;
}
