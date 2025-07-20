#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int i, j, k, l;
  string s;
  cin >> k;
  for (i = 0; i < k; i++) {
    cin >> s;
    l = s.length();
    if (l <= 10) {
      cout << s << endl;
    } else {
      cout << s[0] << l - 2 << s[l - 1] << endl;
    }
  }
  return 0;
}
