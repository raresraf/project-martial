#include <bits/stdc++.h>
#include <cstring>
#define fastIO                                                                 \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);

using namespace std;

int main() {
  fastIO int t;
  long int str, n;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    str = s.size();
    if (str > 10) {
      n = str - 2;
      cout << s.front() << n << s.back() << endl;
    } else
      cout << s << endl;
  }
  return 0;
}
