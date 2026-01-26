#include <bits/stdc++.h>
using namespace std;
map<int, int> m;
int main() {
  int n, tot = 0;
  cin >> n;
  while (n--) {
    int a, b, c;
    cin >> a >> b >> c;
    m[a]++;
    m[c]++;
    m[b]++;
    if (m[1] >= 2)
      tot++;
    m.clear();
  }
  cout << tot << "\n";
}
