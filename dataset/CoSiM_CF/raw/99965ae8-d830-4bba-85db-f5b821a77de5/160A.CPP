#include <bits/stdc++.h>
using namespace std;

int minimum_coins(int a, int b, vector<int> c) {
  int ii = c.size() - 1;
  int aux = 1;
  while (a >= b) {
    a -= c[ii - aux];
    b += c[ii - aux];
    aux++;
  }
  return aux;
}

int main() {
  int N;
  scanf("%d", &N);

  vector<int> v;

  int a;
  while (N--) {
    scanf("%d", &a);
    v.push_back(a);
  }

  sort(v.begin(), v.end());

  int sum2 = v[v.size() - 1];
  int sum1 = 0;

  for (int i = 0; i < v.size() - 1; i++) {
    sum1 += v[i];
  }

  int ans = minimum_coins(sum1, sum2, v);
  cout << ans;

  return 0;
}
