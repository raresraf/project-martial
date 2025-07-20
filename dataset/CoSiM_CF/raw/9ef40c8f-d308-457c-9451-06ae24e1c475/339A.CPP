#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  string s = "";
  cin >> s;
  int temp = 0;
  vector<int> digits(0);
  for (int i = 0; i < s.length(); i++) {
    if (s[i] != '+') {
      digits.push_back(int(s[i]));
    }
  }

  if (digits.size() < 2) {
    cout << s << endl;
    return 0;
  } else {
    for (int i = 0; i < digits.size() - 1; i++) {
      for (int j = 0; j < digits.size() - i - 1; j++) {
        if (digits[j] > digits[j + 1]) {
          temp = digits[j];
          digits[j] = digits[j + 1];
          digits[j + 1] = temp;
        }
      }
    }
  }

  for (int i = 0; i < digits.size(); i++) {
    if (i <= digits.size() - 2)
      cout << (char)digits[i] << '+';
    else
      cout << (char)digits[i];
  }

  return 0;
}