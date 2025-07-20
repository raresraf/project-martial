#include <bits/stdc++.h>
using namespace std;
#define M 1000000007
typedef long long ll;

void gz() {

#ifndef ONLINE_JUDGE
  freopen("peepee.txt", "r", stdin);
  freopen("poopoo.txt", "w", stdout);
#endif

  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.precision(10);
}

// int p = 31;

int main(int argc, char const *argv[]) {
  gz();
  {
    int t;
    cin >> t;
    while (t--) {
      string s;
      cin >> s;
      if (s.length() >= 12) {
        string x;
        x += s[0];
        x += to_string(s.length() - 2);
        x += s[s.length() - 1];
        cout << x << endl;
      } else
        cout << s << endl;
    }
  }
  return 0;
}