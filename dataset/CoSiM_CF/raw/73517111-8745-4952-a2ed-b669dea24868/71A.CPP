#include <bits/stdc++.h>
using namespace std;

#define endl "\n"
#define ends return 0
#define ll long long
#define fast                                                                   \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);
#define test                                                                   \
  int t;                                                                       \
  cin >> t;                                                                    \
  while (t--)

void run() {

  test {
    string s;
    cin >> s;
    ll l = s.length();

    if (l <= 10)
      cout << s << endl;
    else {
      cout << s[0] << l - 2 << s[l - 1] << endl;
    }
  }
}
int main() {
  fast;
  run();
  ends;
}