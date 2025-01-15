#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef long long ll;
typedef vector<ll> vll;
typedef pair<int, int> tupl;
typedef pair<int, pair<int, int>> tripl;

int main() {
  ll k, n, w;
  scanf("%lld %lld %lld", &k, &n, &w);
  ll totalSum = (w * (w + 1)) / 2 * k;
  if (totalSum > n)
    printf("%lld", totalSum - n);
  else
    printf("0");
}
