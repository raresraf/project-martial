#include <bits/stdc++.h>

using namespace std;

int money[101];

int main() {

  int n;
  int sum = 0;

  scanf("%d", &n);

  for (int i = 0; i < n; ++i) {
    scanf("%d", money + i);
    sum += money[i];
  }

  sort(money, money + n);

  int count = 0;
  int till_sum = 0;

  for (int i = n - 1; i >= 0; --i) {

    till_sum += money[i];
    count++;

    sum -= money[i];

    // cout << sum << "    " << till_sum << endl;

    if (till_sum > sum)
      break;
  }

  printf("%d\n", count);
}