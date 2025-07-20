#include <bits/stdc++.h>
using namespace std;

#define sz(v) ((int)v.size())
#define ll long long
#define all(v) (v.begin()), (v.end())
#define rall(v) (v.rbegin()), (v.rend())

int main() {
  // freopen("input.txt","r",stdin);
  int n, a, b;
  cin >> n >> a >> b;

  for (int x = 0; x * a <= n; x++) {
    int rem = n - x * a;
    if (rem % b == 0) {
      puts("YES");
      printf("%d %d\n", x, rem / b);
      return 0;
    }
  }

  puts("NO");
  return 0;
}
