#include <algorithm>
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, ara[110], i = 1, j = 1, k, count1 = 1, count2 = 1, min = 0, sumI = 0,
                   sumJ = 0, flag = 0;

  bool found = false;

  cin >> n;

  for (i = 0; i < n; i++) {
    cin >> ara[i];
  }

  if (n == 2 && (ara[0] == ara[1])) {
    cout << "2" << endl;
    return 0;
  }

  sort(ara, ara + n);

  sumI = ara[n - 1];
  sumJ = ara[0];

  i = n - 1;

  if (sumI == sumJ) {
    min = (n / 2) + 1;
    cout << min << endl;
    return 0;
  }

  if (sumI > sumJ) {
    sumJ = 0;
    for (j = 0; j < i; j++)
      sumJ += ara[j];

    if (sumJ < sumI) {
      cout << count1 << endl;
      found = false;
      return 0;
    } else if ((sumJ > sumI) || (sumI == sumJ)) {

      while (1) {
        sumJ = 0;
        i--;
        sumI += ara[i];
        count1++;

        for (j = 0; j < i; j++) {
          sumJ += ara[j];
        }

        if ((sumI > sumJ))
          break;
      }

      if (sumI > sumJ) {
        min = n - i;
        cout << min << endl;
        return 0;
      }
    }
  }
}
