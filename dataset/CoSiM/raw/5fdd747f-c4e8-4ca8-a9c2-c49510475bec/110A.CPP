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
  long int n, i, j, cnt = 0, t;
  cin >> n;
  vector<int> R(n, 0), C(n, 0);

  for (i = 0; i < n; i++)
    for (j = 0; j < n && cin >> t; j++)
      R[i] += t, C[j] += t;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      if (R[i] > C[j])
        cnt++;

  cout << cnt << "\n";
  return 0;
}
