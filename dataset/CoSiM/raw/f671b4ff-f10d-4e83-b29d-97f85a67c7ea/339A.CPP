#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string s;
  int c1 = 0, c2 = 0, c3 = 0, i;
  cin >> s;
  for (i = 0; i < s.length(); i++) {
    if (s[i] == '1') {
      c1++;
    }
    if (s[i] == '2') {
      c2++;
    }
    if (s[i] == '3') {
      c3++;
    }
  }
  for (i = 0; i < s.length(); i += 2) {
    if (c1 > 0) {
      s[i] = '1';
      c1--;
    } else if (c2 > 0) {
      s[i] = '2';
      c2--;
    } else if (c3 > 0) {
      s[i] = '3';
      c3--;
    }
  }
  cout << s << endl;
  return 0;
}
