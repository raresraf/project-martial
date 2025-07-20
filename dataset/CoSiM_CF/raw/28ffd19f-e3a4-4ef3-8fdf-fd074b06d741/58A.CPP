#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int i = 0, c = 0, st = 0;
  cin >> s;
  string a = "hello";
  for (i = 0; i < s.size(); i++) {
    if (s[i] == a[c]) {
      st++;
      c++;
    }
    if (c > 4) {
      break;
    }
  }
  if (st == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
