#include <bits/stdc++.h>
using namespace std;
int main(int argc, char const *argv[]) {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  while (t--) {
    for (int i = 0; i < n - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G')
        s[i] = 'G', s[i + 1] = 'B', i++;
    }
    // cout << s << endl ;
  }
  cout << s << endl;
  return 0;
}