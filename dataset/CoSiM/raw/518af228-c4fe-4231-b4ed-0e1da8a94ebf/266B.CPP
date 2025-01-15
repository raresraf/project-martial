// 266B
#include <algorithm>
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int n, t;
  string s;
  cin >> n >> t;
  cin >> s;
  int x = 1;
  while (t--) {
    for (int i = 0; i < n; i += x) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        x = 2;
      } else
        x = 1;
    }
  }
  cout << s;
  return 0;
}
