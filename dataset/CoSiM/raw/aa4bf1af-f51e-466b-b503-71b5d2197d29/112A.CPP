#include <bits/stdc++.h>
using namespace std;

int main() {
  string s, s2;
  bool a = false, b = false;
  cin >> s;
  cin >> s2;
  int t1 = 0, t2 = 0;
  int size = s.size();
  for (int i = 0; i < size; i++) {
    if (tolower(s[i]) < tolower(s2[i])) {
      a = true;
      break;
    } else if (tolower(s2[i]) < tolower(s[i])) {
      b = true;
      break;
    }
  }
  if (a == true)
    cout << -1 << endl;
  else if (b == true)
    cout << 1 << endl;
  else
    cout << 0 << endl;
}
