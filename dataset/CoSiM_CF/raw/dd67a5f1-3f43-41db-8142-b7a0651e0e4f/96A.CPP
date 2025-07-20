#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;

  int count = 1;
  int zcount = 1;
  int myone, myzero;
  for (int i = 0; i < s.length() - 1; i++) {
    if (s[i] == '1' && s[i] == s[i + 1]) {
      count++;
      myone = count;
    } else if (s[i] == '0' && s[i] == s[i + 1]) {
      zcount++;
      myzero = zcount;
    } else {
      count = 1;
      zcount = 1;
    }
  }
  if (myone >= 7 || myzero >= 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
