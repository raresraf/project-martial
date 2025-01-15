#include <iostream>
#include <string>
using namespace std;
int main() {
  int n, t, i;
  string s;
  cin >> n >> t >> s;
  while (t--) {
    for (i = 0; i < n; i++) {
      if (s[i] == 'G' && s[i - 1] == 'B') {
        s[i - 1] = 'G';
        s[i] = 'B';
        ++i;
      }
    }
  }
  cout << s << endl;
  return 0;
}
