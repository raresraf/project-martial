#include <bits/stdc++.h>
using namespace std;

int main() {

  long long int n, cnt = 0;
  bool lucky = true;
  cin >> n;
  while (n > 0) {
    if (n % 10 == 4 || n % 10 == 7)
      cnt++;
    n /= 10;
  }
  while (cnt > 0) {
    if (cnt % 10 != 4 && cnt % 10 != 7) {
      lucky = false;
      break;
    }
    cnt /= 10;
  }

  if (lucky)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
