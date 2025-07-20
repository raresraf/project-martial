#include <bits/stdc++.h>
using namespace std;

int main() {
  long long int n, counter = 0;

  cin >> n;

  while (n != 0) {
    int temp = n % 10;

    if (temp == 7 || temp == 4) {
      counter++;
    }
    n = n / 10;
  }
  if (counter == 7 || counter == 4)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
/**
40047
output:
NO

7747774
output:
YES

1000000000000000000
output:
NO
**/
