#include <bits/stdc++.h>

using namespace std;

int main() {
  long long int a, count1 = 0;
  int temp;

  // scanf("%lld",&a);
  cin >> a;
  while (1) {
    if (a == 0) {
      break;
    }
    temp = a % 10;
    if (temp == 4 || temp == 7) {
      count1++;
    }
    a = a / 10;
  }
  if (count1 == 4 || count1 == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
