#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  int count = 0;
  char last = 'x';
  cin >> s;
  for (char const &c : s) {
    if (c != last) {
      count = 1;
      last = c;
    } else {
      if (++count == 7) {
        cout << "YES";
        return 0;
      }
    }
  }
  cout << "NO";
}
