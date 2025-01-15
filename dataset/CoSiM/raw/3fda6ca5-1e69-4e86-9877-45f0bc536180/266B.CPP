#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t;
  string s;
  while (cin >> n >> t) {
    cin >> s;
    while (t--) {

      for (int i = 0; i < n; i++) {
        for (int k = i; k < n; k++) {
          if (s[k] == 'B' && s[k + 1] == 'G') {
            swap(s[k], s[k + 1]);
            k++;
          }
        }
      }
    }

    cout << s << endl;
  }

  return 0;
}
