/************************************************\
*                                                *
* Submission By Deepanshu Luhach aka DragonFist  *
* Institution Name: Army Institute of Technology *
*                                                *
\************************************************/
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;
typedef vector<long long int> vl;
#define PI 3.14159265359

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  int n, k, c, cnt = 0, i;
  cin >> n >> k;
  int a[n] = {0};
  for (i = 0; i < n; i++) {
    cin >> a[i];
    if (i == k - 1)
      c = a[i];
  }
  sort(a, a + n);
  i = n - 1;
  // cout<<i<<" "<<c<<endl;
  while (a[i] >= c && a[i] > 0 && i >= 0) {
    cnt++;
    i--;
  }
  cout << cnt;
}
