#include <bits/stdc++.h>
#define ll long long int
using namespace std;
int main() {
  int n, k;
  cin >> n;
  while (n--) {
    string s;
    cin >> s;

    if (s.size() <= 10)
      cout << s << endl;
    else
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
  }
}
