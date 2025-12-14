#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s, s1;
  int i, j;
  while (cin >> s) {
    s1 = s;
    if (s1[0] >= 'a' && s1[0] <= 'z') {
      s1[0] = toupper(s1[0]);
      cout << s1 << endl;
    } else {
      cout << s << endl;
    }
  }
  return 0;
}
