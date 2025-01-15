#include <bits/stdc++.h>
using namespace std;
#define pb push_back
#define pf push_front
#define ll long long

int main() {
  // freopen("input.txt", "r", stdin);
  // freopen("output.txt", "w", stdout);

  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n;
  int i = 1, j = 2, k = 3;
  int sum = 0;

  cin >> n;

  for (; k <= n; j++, k++) {
    sum += (i * j * k);
  }

  printf("%d\n", sum);

  return 0;
}