#include <cstdio>
#include <iostream>

using namespace std;

int main() {

  int n, t, i;
  string s;

  cin >> n >> t;
  cin >> s;

  while (t--) {
    for (i = 0; i < s.size() - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        i++;
      }
    }
  }
  cout << s << endl;

  return 0;
}
