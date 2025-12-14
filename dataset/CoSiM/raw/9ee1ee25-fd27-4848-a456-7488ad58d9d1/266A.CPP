#include <bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define mp make_pair
#define pb push_back
typedef long long ll;
int main() {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(false);
  int n, ans = 0;
  cin >> n;
  string xs;
  cin >> xs;
  bool found = false;
  char init = xs[0];
  for (int i = 0; i < xs.size(); i++) {
    int count = 0;
    init = xs[i];
    bool inwhile = false;
    while (xs[i] == init) {
      ++count;
      i++;
      inwhile = true;
    }
    if (count > 1)
      ans += (count - 1);
    if (inwhile)
      --i;
  }
  cout << ans << endl;
}