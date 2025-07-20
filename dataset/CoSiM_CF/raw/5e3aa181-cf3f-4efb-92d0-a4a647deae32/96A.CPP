#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  string input;
  int counter = 0;
  cin >> input;

  if (input.empty()) {
    return -1;
  }

  char key = input[0];
  for (auto chr : input) {
    if (chr == key) {
      ++counter;
    } else {
      key = chr;
      counter = 1;
    }

    if (counter == 7) {
      cout << "YES";
      return 0;
    }
  }
  cout << "NO";
  return 0;
}
