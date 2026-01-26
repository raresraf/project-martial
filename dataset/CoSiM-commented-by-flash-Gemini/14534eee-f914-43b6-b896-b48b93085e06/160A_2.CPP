#include <algorithm>
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, ara[110], count = 0, sum1 = 0, sum2 = 0, j, i;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> ara[i];
  }
  sort(ara, ara + n);
  int k = n - 1;
  while (1) {
    for (i = n - 1; i >= k; i--) {
      sum1 += ara[i];
      count++;
    }
    for (j = 0; j < k; j++) {
      sum2 += ara[j];
    }
    if (sum1 > sum2) {
      break;
    } else {
      k--;
      count = 0;
      sum1 = 0;
      sum2 = 0;
    }
  }
  cout << count << endl;
}
