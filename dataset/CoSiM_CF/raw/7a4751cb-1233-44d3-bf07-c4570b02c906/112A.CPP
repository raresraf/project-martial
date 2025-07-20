#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
using namespace std;

int main() {
  // freopen("A.txt","r",stdin);
  int i, n, x, y, l, r, d, u, j, cnt = 0;
  cin >> n;
  vector<pair<int, int>> A;
  for (i = 0; i < n && cin >> x >> y; i++)
    A.push_back(make_pair(x, y));

  for (i = 0; i < n; i++) {
    l = r = d = u = 0;
    for (j = 0; j < n; j++) {

      if (A[i].first > A[j].first && A[i].second == A[j].second)
        r++;
      else if (A[i].first < A[j].first && A[i].second == A[j].second)
        l++;
      else if (A[i].first == A[j].first && A[i].second < A[j].second)
        d++;
      else if (A[i].first == A[j].first && A[i].second > A[j].second)
        u++;
    }
    // cout<<l<<" "<<r<<" "<<d<<" "<<u<<"\n";
    if (l * r * d * u)
      cnt++;
  }
  cout << cnt << "\n";
  return 0;
}
