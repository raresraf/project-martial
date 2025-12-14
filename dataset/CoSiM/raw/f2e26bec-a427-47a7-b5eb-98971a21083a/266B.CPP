#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int n;
  int t;
  cin >> n >> t;
  string s;
  cin >> s;
  while (t--) {
    int i = 0;
    while (i <= n - 1) {
      if (s[i] == 'B' and s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        i += 2;
      } else
        i += 1;
    }
  }
  cout << s << '\n';
  return 0;
}