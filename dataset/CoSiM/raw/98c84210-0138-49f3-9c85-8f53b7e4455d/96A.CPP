#include <iostream>
#include <string>

using namespace std;
bool is_dangerous(string &);

int main() {
  string input;
  cin >> input;

  if (is_dangerous(input))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}

bool is_dangerous(string &input) {
  int length = input.length(), i = 0, is_dangerous = false;

  while (i < length - 1) {
    int count = 0;
    // if two consecutive charecters are same
    if (input.at(i) == input.at(i + 1)) {
      int j = i;
      char digit = input.at(i);

      count = 0;

      // find out the number of consecutive charecters
      while (j < length && input.at(j) == digit) {
        count++;
        j++;
      }
    }

    if (count >= 7) {
      is_dangerous = true;
      break;
    }

    i++;
  }

  return is_dangerous;
}