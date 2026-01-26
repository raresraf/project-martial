#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, x = 0;
  cin >> n;
  while (n--) {
    string s;
    cin >> s;
    int l = s.length();
    for (int i = 0; i < l; i++) {
      if (s[i] == '+') {
        x++;
        break;

      } else if (s[i] == '-') {
        x--;
        break;
      }
    }
    // cout<<" => x="<<x<<endl;
  }
  cout << x;
  return 0;
}
