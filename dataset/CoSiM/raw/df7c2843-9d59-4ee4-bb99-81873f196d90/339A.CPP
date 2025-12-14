#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  string input = "";
  string s = "";
  string x = "";
  string result = "";
  cin >> input;
  for (int i = 0; i < input.length(); i++) {
    if (input[i] != '+') {
      s += input[i];
    }
  }

  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1') {
      x += s[i];
      x += '+';
    }
  }
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '2') {
      x += s[i];
      x += '+';
    }
  }
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '3') {
      x += s[i];
      x += '+';
    }
  }

  for (int i = 0; i < x.length(); i++) {
    result = x.substr(0, x.length() - 1);
  }

  cout << result;

  return 0;
}
