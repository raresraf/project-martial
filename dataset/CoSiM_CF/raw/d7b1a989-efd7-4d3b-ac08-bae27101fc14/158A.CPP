#include <bits/stdc++.h>

using namespace std;

int main() {
  int a, k, n, temp;
  vector<int> v;
  int count1 = 0;
  scanf("%d %d", &a, &k);
  for (int i = 1; i <= a; i++) {
    scanf("%d", &n);

    v.push_back(n);

    if (i == k) {
      temp = n;
    }
  }

  for (int j = 0; j < a; j++) {

    if (v[j] != 0 && v[j] >= temp) {
      count1++;
    }
  }
  printf("%d\n", count1);

  return 0;
}
