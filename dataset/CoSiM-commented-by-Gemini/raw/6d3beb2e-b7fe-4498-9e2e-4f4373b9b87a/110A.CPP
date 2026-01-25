#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  int sz = s.length();
  int cnt = 0;
  for (int i = 0; i < sz; ++i) {
    if (s[i] == '4' || s[i] == '7')
      cnt++;
  }

  if (cnt == 4 || cnt == 7)
    cout << "YES\n";
  else
    cout << "NO\n";

  return 0;
}
