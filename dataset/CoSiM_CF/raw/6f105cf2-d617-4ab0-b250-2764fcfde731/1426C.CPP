/****15-Sept-2020*****/
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pb push_back
#define mp make_pair
#define ss set<string> // insert() .top()
#define si set<int>
#define pii pair<int, int>
#define vi vector<int> // push_back()
#define mii map<int, int>
#define pq priority_queue<int> // insert()
#define mod 1000000007
#define inf 1e18
#define ps(x, y) fixed << setprecision(y) << x
#define m(arr, n, type) type *arr = new type[n];
#define TestCase(x)                                                            \
  int x;                                                                       \
  cin >> x;                                                                    \
  while (x--)
#define FOR for (int i = 0; i < n; i++)
#define gcd(a, b) __gcd(a, b)
#define lcm(a, b) ((a) * ((b) / gcd(a, b)))
#define Mamun                                                                  \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0)
/*erase(start_index,delete_length)*/
/*int myfun(int a,int b)
{
    return a>b;
}
*/
/*sort(vect.rbegin(),vect.rend())*/
ll ans, res, p, q;
void solve() {
  int n;
  cin >> n;
  int i = 1;
  ans = 0;
  for (int i = 1; i <= n / i; i++)
    res = i;
  if (n % res)
    cout << res + n / res - 1 << "\n";
  else
    cout << res + n / res - 2 << "\n";
}
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  int tt = 1;
  cin >> tt;
  while (tt--)
    solve();
}
