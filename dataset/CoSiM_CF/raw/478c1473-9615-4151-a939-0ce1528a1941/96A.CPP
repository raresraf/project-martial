#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int ans = 1, d = 0;
  char l = '0';

  cin >> s;

  for (char c : s) {
    if (c == l) {
      ++d;
      ans = max(d, ans);
    } else {
      l = c;
      d = 1;
    }
  }
  if (ans >= 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
