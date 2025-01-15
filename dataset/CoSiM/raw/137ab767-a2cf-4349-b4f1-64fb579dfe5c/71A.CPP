#include <iostream>
#include <string>

// http://codeforces.com/problemset/problem/71/A
// A. Way Too Long Words

using namespace std;

int main() {
  int n; // int data type accept characters but when u cout it it will output 0;
  string s; // string data type accept characters and numbers as well;
  cin >> n;
  while (n--) {
    cin >> s;
    if (s.length() > 10) {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      cout << s << endl;
    }
  }
  return 0;
}