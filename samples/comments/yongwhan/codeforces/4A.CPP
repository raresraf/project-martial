#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  int w;
  cin >> w;
  cout << (w % 2 == 0 && w > 2 ? "YES" : "NO") << endl;
  return 0;
}
