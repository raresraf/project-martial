/**
 * @file 339A.CPP
 * @brief Sorts the terms in a mathematical sum represented as a string.
 *
 * The input is a string representing a sum of single-digit numbers (1, 2, or 3),
 * e.g., "3+2+1". The program outputs the sum with the terms rearranged in
 * non-decreasing order, e.g., "1+2+3".
 *
 * @algorithm Counting Sort. The program first counts the occurrences of each number
 * ('1', '2', '3'). Then, it reconstructs the string by printing the numbers in
 * sorted order, separated by plus signs.
 *
 * @time_complexity O(N), where N is the length of the input string, as the process
 * involves a single pass to count and another pass to print.
 * @space_complexity O(N) to store the input string. The counting variables use O(1) space.
 */
#include <bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define mp make_pair
#define pb push_back
typedef long long ll;
int main() {
  // freopen("input.txt", "r", stdin);
  // Optimizes I/O operations by decoupling C++ streams from C standard streams.
  ios_base::sync_with_stdio(false);
  string s;
  cin >> s;
  int count = 0;
  int one = 0, two = 0, three = 0;
  /**
   * @block_logic This loop iterates through the input string to count the
   * occurrences of each number. The '+' characters are implicitly ignored.
   * `count` tracks the total number of digits.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '1') {
      ++count;
      ++one;
    }
    if (s[i] == '2') {
      ++count;
      ++two;
    }
    if (s[i] == '3') {
      ++count;
      ++three;
    }
  }
  /**
   * @block_logic This section reconstructs and prints the sorted sum.
   * It iterates through the counts of '1's, then '2's, then '3's.
   * The `count` variable is used to determine if a '+' should be printed
   * or if it is the last term in the sum.
   */
  for (int i = 0; i < one; i++) {
    count -= 1;
    if (count == 0)
      cout << "1" << endl;
    else
      cout << "1+";
  }
  for (int i = 0; i < two; i++) {
    count -= 1;
    if (count == 0)
      cout << "2" << endl;
    else
      cout << "2+";
  }
  for (int i = 0; i < three; i++) {
    count -= 1;
    if (count == 0)
      cout << "3" << endl;
    else
      cout << "3+";
  }
}