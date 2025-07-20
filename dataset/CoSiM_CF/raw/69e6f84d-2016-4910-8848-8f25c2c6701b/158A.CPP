#include <bits/stdc++.h>
using namespace std;
#define gc getchar_unlocked
#define fo(i, n) for (i = 0; i < n; i++)
#define Fo(i, k, n) for (i = k; k < n ? i < n : i > n; k < n ? i += 1 : i -= 1)
#define ll long long
#define si(x) scanf("%d", &x)
#define sl(x) scanf("%lld", &x)
#define ss(s) scanf("%s", s)
#define pi(x) printf("%d\n", x)
#define pl(x) printf("%lld\n", x)
#define ps(s) printf("%s\n", s)
#define deb(x) cout << #x << "=" << x << endl
#define deb2(x, y) cout << #x << "=" << x << "," << #y << "=" << y << endl
#define pb push_back
#define mp make_pair
#define clr(x) memset(x, 0, sizeof(x))
#define PI 3.1415926535897932384626
typedef pair<int, int> pii;
typedef pair<ll, ll> pl;
typedef vector<int> vi;
typedef vector<ll> vl;
typedef vector<pii> vpii;
typedef vector<pl> vpl;
typedef vector<vi> vvi;
typedef vector<vl> vvl;

const int mod = 1'000'000'007;
const int N = 3e5, M = N;
//=======================

void solve() {
  int i, n, k, x, *a = NULL, count = 0;

  cin >> n >> k;

  a = new int[n];

  for (i = 0; i < n; i++) {
    cin >> a[i];
  }

  x = a[k - 1];

  for (i = 0; i < n; i++) {
    if (a[i] >= x && (x != 0 || a[i] != 0)) {
      ++count;
    } else if (a[i] < x) {
      break;
    }
  }

  cout << count;

  delete[] a;
}

int main() {
  ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0);

  solve();

  return 0;
}
