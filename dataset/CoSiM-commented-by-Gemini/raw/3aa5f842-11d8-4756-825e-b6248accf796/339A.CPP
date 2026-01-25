#include <iostream>
#include <string>
using namespace std;

int main() {
  string s, str;
  cin >> s;
  for (auto c : s) {
    if (c == '1') {
      str += c;
      str += '+';
    }
  }
  for (auto c : s) {
    if (c == '2') {
      str += c;
      str += '+';
    }
  }
  for (auto c : s) {
    if (c == '3') {
      str += c;
      str += '+';
    }
  }
  str.erase(str.length() - 1, 1);
  cout << str << endl;
  return 0;
}
