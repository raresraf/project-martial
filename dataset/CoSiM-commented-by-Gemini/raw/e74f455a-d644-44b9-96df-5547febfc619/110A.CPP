/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem.
 *
 * A number is "lucky" if it consists only of the digits 4 and 7.
 * This program determines if a given number is "nearly lucky", which means
 * the count of lucky digits ('4' or '7') within the number is itself a lucky number.
 *
 * @algorithm
 * 1. Read the input number as a string.
 * 2. Count the occurrences of '4' and '7' in the string.
 * 3. Convert this count into a string.
 * 4. Check if the count string is composed solely of '4's and '7's.
 * 5. If so, the original number is "nearly lucky".
 *
 * @time_complexity O(L), where L is the number of digits in the input number.
 * @space_complexity O(L) to store the input string.
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

using namespace std;

// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
// using namespace __gnu_pbds;
// typedef tree<int,null_type,less<int>,rb_tree_tag,
// tree_order_statistics_node_update> ordered_set;

// Unused global arrays.
int a[105], b[105];
int main() {
  string s;
  int i, cp = 0;
  cin >> s;
  int sz = s.length();
  /**
   * @block_logic Counts the number of lucky digits ('4' or '7') in the input string.
   * The total count is stored in the variable `cp`.
   */
  for (i = 0; i < sz; ++i) {
    if (s[i] == '4' || s[i] == '7') {
      cp++;
    }
  }

  // Convert the count of lucky digits to a string for inspection.
  string str = to_string(cp);

  /**
   * @block_logic Checks if the count (`cp`) is itself a lucky number.
   * It iterates through the digits of the count string. If any digit is
   * not a '4' or a '7', the loop breaks.
   */
  for (i = 0; i < str.length(); ++i) {
    if (str[i] == '4' || str[i] == '7') {
      continue;
    } else
      break;
  }

  /**
   * @block_logic If the loop completed without breaking, it means all digits
   * in the count were lucky digits.
   */
  if (i == str.length())
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}