#include <iostream>
#include <string>

using namespace std;

/**
 * @brief This program determines if a given binary string contains a sequence of seven identical digits.
 *
 * It reads a binary string from the standard input, searches for the substrings "1111111" or "0000000",
 * and prints "YES" if either is found, otherwise it prints "NO".
 */
int main() {
  size_t n;
  bool is_found = false;
  string pos;
  cin >> pos;
  n = pos.find("1111111");
  if (n >= 0 && n < pos.size()) {
    is_found = true;
  } else {
    n = pos.find("0000000");
    if (n >= 0 && n < pos.size()) {
      is_found = true;
    }
  }
  if (is_found) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}