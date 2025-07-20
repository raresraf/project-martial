#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  string input;
  cin >> input;

  size_t pos = 0;
  while (pos < input.length() && input[pos] != 'h') {
    ++pos;
  }
  ++pos;
  while (pos < input.length() && input[pos] != 'e') {
    ++pos;
  }
  ++pos;
  while (pos < input.length() && input[pos] != 'l') {
    ++pos;
  }
  ++pos;
  while (pos < input.length() && input[pos] != 'l') {
    ++pos;
  }
  ++pos;
  while (pos < input.length() && input[pos] != 'o') {
    ++pos;
  }

  if (pos < input.length()) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
