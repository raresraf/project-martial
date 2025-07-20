#include <algorithm>
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, x, count = 0, sum = 0, sum1 = 0;

  cin >> n;

  vector<int> a;

  for (int i = 0; i < n; i++) {
    cin >> x;
    sum = sum + x;
    a.push_back(x);
  }

  sort(a.begin(), a.end());

  //    for(int i=0; i<n; i++)
  //    {
  //        sum=sum+a[i];
  //    }

  while (1) {
    sum1 = sum1 + a[n - 1];

    int s = sum - sum1;

    count++;

    if (s < sum1) {
      break;
    }

    n--;
  }

  cout << count;
}
