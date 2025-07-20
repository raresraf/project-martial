#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int c1 = 0, c2 = 0, c3 = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1') {
      c1++;
    } else if (s[i] == '2') {
      c2++;
    } else if (s[i] == '3') {
      c3++;
    }
  }
  int len = s.length();
  s.erase(0, len);

  for (int i = 0; i < c1; i++) {
    s.push_back('1');
    s.push_back('+');
  }
  for (int i = 0; i < c2; i++) {
    s.push_back('2');
    s.push_back('+');
  }
  for (int i = 0; i < c3; i++) {
    s.push_back('3');
    s.push_back('+');
  }
  s.pop_back();
  cout << s << "\n";
}
