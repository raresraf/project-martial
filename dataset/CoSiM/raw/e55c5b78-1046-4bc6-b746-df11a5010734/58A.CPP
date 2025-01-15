#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;
  int l = s.length();
  string text = "hello";
  int l1 = text.length();
  int pos = 0;
  string ans = "YES";
  for (int i = 0; i < l1; i++) {
    bool flag = false;
    for (int j = pos; j < l; j++) {
      if (s[j] == text[i]) {
        pos = j + 1;
        flag = true;
        break;
      }
    }
    if (!flag) {
      ans = "NO";
      break;
    }
  }
  cout << ans << endl;
  return 0;
}