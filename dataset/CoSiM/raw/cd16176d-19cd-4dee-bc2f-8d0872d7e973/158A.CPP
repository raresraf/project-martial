#include <bits/stdc++.h>

#define ll long long int

#define mod 1000000007

using namespace std;

int main()

{

  int n, k;

  cin >> n >> k;

  vector<int> a(n + 1);

  for (int i = 1; i <= n; i++)

    cin >> a[i];

  int ans = 0;

  for (int i = 1; i <= n; i++)

  {

    if (a[i] >= a[k] && a[i] > 0)

      ans++;
  }

  cout << ans;
}