
//													\[T]/
//Praise
//													 (*)
//The 													 / \  Sun

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef long double dbl;
typedef vector<int> vi;
typedef pair<int, int> pii;
typedef pair<long long, long long> pll;
typedef vector<long long> vl;
typedef vector<pair<int, int>> vii;
#define mp make_pair
#define pb push_back
#define fr(i, j, k) for (int i = j; i < k; i++)
#define FR(i, j, k) for (int i = j; i <= 0; i--)
#define MOD 1000000007
#define all(v) v.begin(), v.end()

void fast() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
}

int main() {
  int n, sum = 0;
  cin >> n;
  vi a(n + 1);
  vi dp(n + 1);
  a[0] = 0;
  fr(i, 1, n + 1) {
    cin >> a[i];
    sum += a[i];
  }
  sort(all(a));
  reverse(a.begin() + 1, a.end());

  //	fr(i, 1, n+1)
  //	{
  //		cout << a[i];
  //	}
  //	cout << "\n";

  dp[0] = 0;
  fr(i, 1, n + 1) {
    dp[i] = dp[i - 1] + a[i];
    if (dp[i] > sum - dp[i]) {
      cout << i;
      return 0;
    }
  }
}
